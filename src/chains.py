# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import requests
from traceback import print_exc
from typing import Any, Iterable
from typing import Dict
from typing import Generator
from typing import List

from langchain_nvidia_ai_endpoints.callbacks import get_usage_callback
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableAssign
from langchain_core.runnables import RunnablePassthrough
from requests import ConnectTimeout

from .base import BaseExample
from .utils import create_vectorstore_langchain
from .utils import get_config
from .utils import get_embedding_model
from .utils import get_llm
from .utils import get_prompts
from .utils import get_ranking_model
from .utils import get_text_splitter
from .utils import get_vectorstore
from .utils import format_document_with_source
from .utils import streaming_filter_think, get_streaming_filter_think_parser
from .reflection import ReflectionCounter, check_context_relevance, check_response_groundedness
from .utils import normalize_relevance_scores

logger = logging.getLogger(__name__)
VECTOR_STORE_PATH = "vectorstore.pkl"
TEXT_SPLITTER = None
settings = get_config()
document_embedder = get_embedding_model(model=settings.embeddings.model_name, url=settings.embeddings.server_url)
ranker = get_ranking_model(model=settings.ranking.model_name, url=settings.ranking.server_url, top_n=settings.retriever.top_k)
query_rewriter_llm_config = {"temperature": 0.7, "top_p": 0.2, "max_tokens": 1024}
logger.info("Query rewriter llm config: model name %s, url %s, config %s", settings.query_rewriter.model_name, settings.query_rewriter.server_url, query_rewriter_llm_config)
query_rewriter_llm = get_llm(model=settings.query_rewriter.model_name, url=settings.query_rewriter.server_url, **query_rewriter_llm_config)
prompts = get_prompts()
vdb_top_k = int(os.environ.get("VECTOR_DB_TOPK", 40))

try:
    VECTOR_STORE = create_vectorstore_langchain(document_embedder=document_embedder)
except Exception as ex:
    VECTOR_STORE = None
    logger.info("Unable to connect to vector store during initialization: %s", ex)

# Get a StreamingFilterThinkParser based on configuration
StreamingFilterThinkParser = get_streaming_filter_think_parser()

class APIError(Exception):
    """Custom exception class for API errors."""
    def __init__(self, message: str, code: int = 400):
        logger.error("APIError occurred: %s with HTTP status: %d", message, code)
        print_exc()
        self.message = message
        self.code = code
        super().__init__(message)

class UnstructuredRAG(BaseExample):

    def ingest_docs(self, data_dir: str, filename: str, collection_name: str = "", vdb_endpoint: str = "") -> None:
        """Ingests documents to the VectorDB.
        It's called when the POST endpoint of `/documents` API is invoked.

        Args:
            data_dir (str): The path to the document file.
            filename (str): The name of the document file.
            collection_name (str): The name of the collection to be created in the vectorstore.

        Raises:
            ValueError: If there's an error during document ingestion or the file format is not supported.
        """
        try:
            # Load raw documents from the directory
            _path = data_dir
            raw_documents = UnstructuredFileLoader(_path).load()

            if raw_documents:
                global TEXT_SPLITTER  # pylint: disable=W0603
                # Get text splitter instance, it is selected based on environment variable APP_TEXTSPLITTER_MODELNAME
                # tokenizer dimension of text splitter should be same as embedding model
                if not TEXT_SPLITTER:
                    TEXT_SPLITTER = get_text_splitter()

                # split documents based on configuration provided
                logger.info(f"Using text splitter instance: {TEXT_SPLITTER}")
                documents = TEXT_SPLITTER.split_documents(raw_documents)
                vs = get_vectorstore(document_embedder, collection_name, vdb_endpoint)
                # ingest documents into vectorstore
                vs.add_documents(documents)
            else:
                logger.warning("No documents available to process!")

        except ConnectTimeout as e:
            raise APIError(
                "Connection timed out while accessing the embedding model endpoint. Verify server availability.",
                code=504
            ) from e
        except Exception as e:
            if "[403] Forbidden" in str(e) and "Invalid UAM response" in str(e):
                raise APIError(
                    "Authentication or permission error: Verify NVIDIA API key validity and permissions.",
                    code=403
                ) from e
            if "[404] Not Found" in str(e):
                raise APIError(
                    "API endpoint or payload is invalid. Ensure the model name is valid.",
                    code=404
                ) from e
            raise APIError("Failed to upload document. " + str(e), code=500) from e

    @staticmethod
    def flatten_messages(system_prompt: str, chat_history: List[tuple], user_query: str) -> str:
        """
        Combine system prompt, chat history, and user query into a single plain text prompt.
        """
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt.strip())

        for role, content in chat_history:
            if role == "user":
                prompt_parts.append(f"User: {content.strip()}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content.strip()}")

        if user_query:
            prompt_parts.append(f"User: {user_query.strip()}")

        return "\n\n".join(prompt_parts)

    def llm_chain(self, query: str, chat_history: List[Dict[str, Any]], **kwargs) -> Generator[str, None, None]:
        """Execute a simple LLM chain using the components defined above.
        It's called when the `/generate` API is invoked with `use_knowledge_base` set to `False`.
        """
        try:
            logger.info("Using llm to generate response directly without knowledge base.")
            system_prompt = ""
            conversation_history = []
            user_message = []
            
            # Load system prompt template
            system_prompt += prompts.get("chat_template", "")
            
            for message in chat_history:
                if message.role == "system":
                    system_prompt = system_prompt + " " + message.content
                else:
                    conversation_history.append((message.role, message.content))

            if query:
                user_message = [("user", "{question}")]

            model_name = os.getenv("APP_LLM_MODELNAME", "").lower()

            llm = get_llm(**kwargs)

            if model_name.startswith("google"):
                logger.info("Detected Google model - flattening messages for LLM input.")
                final_prompt = self.flatten_messages(system_prompt, conversation_history, query)
                
                chain = llm | StreamingFilterThinkParser | StrOutputParser()
                return chain.stream(final_prompt, config={'run_name': 'llm-stream'})
            else:
                logger.info("Detected non-Google model - using ChatPromptTemplate with roles.")
                message = [("system", system_prompt)] + conversation_history + user_message
                
                self.print_conversation_history(message, query)

                prompt_template = ChatPromptTemplate.from_messages(message)
                chain = prompt_template | llm | StreamingFilterThinkParser | StrOutputParser()
                return chain.stream({"question": query}, config={'run_name': 'llm-stream'})

        except ConnectTimeout as e:
            logger.warning("Connection timed out while making a request to the LLM endpoint: %s", e)
            return iter(["Connection timed out while making a request to the NIM endpoint. Verify if the NIM server is available."])

        except Exception as e:
            logger.warning("Failed to generate response due to exception %s", e)
            print_exc()

            if "[403] Forbidden" in str(e) and "Invalid UAM response" in str(e):
                logger.warning("Authentication or permission error: Verify the validity and permissions of your NVIDIA API key.")
                return iter(["Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."])
            elif "[404] Not Found" in str(e):
                logger.warning("Please verify the API endpoint and your payload. Ensure that the model name is valid.")
                return iter(["Please verify the API endpoint and your payload. Ensure that the model name is valid."])
            else:
                return iter([f"Failed to generate RAG chain response. {str(e)}"])

    def rag_chain(
        self,
        query: str,
        chat_history: List[Dict[str, Any]],
        reranker_top_k: int,
        vdb_top_k: int,
        collection_name: str = "",
        **kwargs,
    ) -> Generator[str, None, None]:
        """Execute a Retrieval Augmented Generation chain using the components defined above.
        It's called when the `/generate` API is invoked with `use_knowledge_base` set to `True`.
        """
        try:
            logger.info("Using rag to generate response from document for the query: %s", query)

            document_embedder = get_embedding_model(model=kwargs.get("embedding_model"), url=kwargs.get("embedding_endpoint"))
            vs = get_vectorstore(document_embedder, collection_name, kwargs.get("vdb_endpoint"))
            if vs is None:
                raise APIError("Vector store not initialized properly. Please check if the vector DB is up and running.", 500)

            llm = get_llm(**kwargs)
            ranker = get_ranking_model(model=kwargs.get("reranker_model"), url=kwargs.get("reranker_endpoint"), top_n=reranker_top_k)
            top_k = vdb_top_k if ranker and kwargs.get("enable_reranker") else reranker_top_k
            logger.info("Setting retriever top k as: %s.", top_k)
            retriever = vs.as_retriever(search_kwargs={"k": top_k})

            system_prompt = ""
            conversation_history = []

            # System prompt
            system_prompt += prompts.get("rag_template", "")

            for message in chat_history:
                if message.role == "system":
                    system_prompt = system_prompt + " " + message.content
                else:
                    conversation_history.append((message.role, message.content))

            model_name = os.getenv("APP_LLM_MODELNAME", "").lower()

            # Get relevant documents
            if os.environ.get("ENABLE_REFLECTION", "false").lower() == "true":
                max_loops = int(os.environ.get("MAX_REFLECTION_LOOP", 3))
                reflection_counter = ReflectionCounter(max_loops)

                context_to_show, is_relevant = check_context_relevance(
                    query, retriever, ranker, reflection_counter
                )

                if not is_relevant:
                    logger.warning("Could not find sufficiently relevant context after maximum attempts")
            else:
                if ranker and kwargs.get("enable_reranker"):
                    logger.info(
                        "Narrowing the collection from %s results and further narrowing it to %s with the reranker for RAG chain.",
                        top_k,
                        reranker_top_k,
                    )
                    logger.info("Setting ranker top n as: %s.", reranker_top_k)

                    context_reranker = RunnableAssign({
                        "context": lambda input: ranker.compress_documents(query=input["question"], documents=input["context"])
                    })

                    retriever = {"context": retriever} | RunnableAssign({"context": lambda input: input["context"]})
                    docs = retriever.invoke(query, config={"run_name": "retriever"})
                    docs = context_reranker.invoke({"context": docs.get("context", []), "question": query}, config={"run_name": "context_reranker"})
                    context_to_show = docs.get("context", [])
                    context_to_show = normalize_relevance_scores(context_to_show)
                else:
                    context_to_show = retriever.invoke(query)

            docs = [format_document_with_source(d) for d in context_to_show]

            if model_name.startswith("google"):
                logger.info("Detected Google model - flattening messages for RAG input.")
                user_query = query
                context_text = "\n\n".join(doc.page_content for doc in docs)
                final_prompt = self.flatten_messages(system_prompt, conversation_history, user_query)
                final_prompt += f"\n\nContext:\n{context_text}"

                chain = llm | StreamingFilterThinkParser | StrOutputParser()
                return chain.stream(final_prompt, config={"run_name": "llm-stream"}), context_to_show
            else:
                logger.info("Detected non-Google model - using ChatPromptTemplate with roles.")
                message = [("system", system_prompt)] + conversation_history + [("user", "{question}")]
                
                self.print_conversation_history(message)

                prompt_template = ChatPromptTemplate.from_messages(message)
                chain = prompt_template | llm | StreamingFilterThinkParser | StrOutputParser()
                return chain.stream({"question": query, "context": docs}, config={"run_name": "llm-stream"}), context_to_show

        except ConnectTimeout as e:
            logger.warning("Connection timed out while making a request to the LLM endpoint: %s", e)
            return iter(["Connection timed out while making a request to the NIM endpoint. Verify if the NIM server is available."]), []

        except Exception as e:
            logger.warning("Failed to generate response due to exception %s", e)
            print_exc()

            if "[403] Forbidden" in str(e) and "Invalid UAM response" in str(e):
                logger.warning("Authentication or permission error: Verify the validity and permissions of your NVIDIA API key.")
                return iter(["Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."]), []
            elif "[404] Not Found" in str(e):
                logger.warning("Please verify the API endpoint and your payload. Ensure that the model name is valid.")
                return iter(["Please verify the API endpoint and your payload. Ensure that the model name is valid."]), []
            else:
                return iter([f"Failed to generate RAG chain response. {str(e)}"]), []


    def rag_chain_with_multiturn(
        self,
        query: str,
        chat_history: List[Dict[str, Any]],
        reranker_top_k: int,
        vdb_top_k: int,
        collection_name: str,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Execute a Retrieval Augmented Generation chain using the components defined above (multi-turn)."""

        try:
            logger.info("Using multiturn rag to generate response from document for the query: %s", query)

            document_embedder = get_embedding_model(model=kwargs.get("embedding_model"), url=kwargs.get("embedding_endpoint"))
            vs = get_vectorstore(document_embedder, collection_name, kwargs.get("vdb_endpoint"))
            if vs is None:
                raise APIError("Vector store not initialized properly. Please check if the vector DB is up and running.", 500)

            llm = get_llm(**kwargs)
            ranker = get_ranking_model(model=kwargs.get("reranker_model"), url=kwargs.get("reranker_endpoint"), top_n=reranker_top_k)
            top_k = vdb_top_k if ranker and kwargs.get("enable_reranker") else reranker_top_k
            logger.info("Setting retriever top k as: %s.", top_k)
            retriever = vs.as_retriever(search_kwargs={"k": top_k})

            # Truncate conversation history if needed
            history_count = int(os.environ.get("CONVERSATION_HISTORY", 15)) * 2 * -1
            chat_history = chat_history[history_count:]

            system_prompt = ""
            conversation_history = []

            system_prompt += prompts.get("rag_template", "")

            for message in chat_history:
                if message.role == "system":
                    system_prompt = system_prompt + " " + message.content
                else:
                    conversation_history.append((message.role, message.content))

            model_name = os.getenv("APP_LLM_MODELNAME", "").lower()

            retriever_query = query
            if chat_history:
                if kwargs.get("enable_query_rewriting"):
                    logger.info("Rewriting query using query rewriter model...")
                    contextualize_q_system_prompt = prompts.get(
                        "query_rewriter_prompt",
                        "Given a chat history and the latest user question, formulate a standalone question."
                    )
                    contextualize_q_prompt = ChatPromptTemplate.from_messages([
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])
                    q_prompt = contextualize_q_prompt | query_rewriter_llm | StreamingFilterThinkParser | StrOutputParser()
                    retriever_query = q_prompt.invoke({"input": query, "chat_history": conversation_history}, config={"run_name": "query-rewriter"})
                    logger.info("Rewritten Query: %s", retriever_query)
                    if retriever_query.replace('"', "'") == "''" or len(retriever_query.strip()) == 0:
                        return iter([""]), []
                else:
                    user_queries = [msg.content for msg in chat_history if msg.role == "user"]
                    retriever_query = ". ".join([*user_queries, query])
                    logger.info("Combined retriever query: %s", retriever_query)

            # Retrieve documents
            if os.environ.get("ENABLE_REFLECTION", "false").lower() == "true":
                max_loops = int(os.environ.get("MAX_REFLECTION_LOOP", 3))
                reflection_counter = ReflectionCounter(max_loops)

                context_to_show, is_relevant = check_context_relevance(
                    retriever_query, retriever, ranker, reflection_counter
                )

                if not is_relevant:
                    logger.warning("Could not find sufficiently relevant context after %d reflection attempts", reflection_counter.current_count)
            else:
                if ranker and kwargs.get("enable_reranker"):
                    logger.info(
                        "Narrowing collection from %s results and further narrowing to %s with reranker for multi-turn rag chain.",
                        top_k,
                        reranker_top_k,
                    )
                    context_reranker = RunnableAssign({
                        "context": lambda input: ranker.compress_documents(query=input["question"], documents=input["context"])
                    })
                    retriever = {"context": retriever} | RunnableAssign({"context": lambda input: input["context"]})
                    docs = retriever.invoke(retriever_query, config={"run_name": "retriever"})
                    docs = context_reranker.invoke({"context": docs.get("context", []), "question": retriever_query}, config={"run_name": "context_reranker"})
                    context_to_show = docs.get("context", [])
                    context_to_show = normalize_relevance_scores(context_to_show)
                else:
                    docs = retriever.invoke(retriever_query, config={"run_name": "retriever"})
                    context_to_show = docs

            docs = [format_document_with_source(d) for d in context_to_show]

            if model_name.startswith("google"):
                logger.info("Detected Google model - flattening multi-turn input for LLM.")
                user_query = query
                context_text = "\n\n".join(doc.page_content for doc in docs)
                final_prompt = self.flatten_messages(system_prompt, conversation_history, user_query)
                final_prompt += f"\n\nContext:\n{context_text}"

                chain = llm | StreamingFilterThinkParser | StrOutputParser()
                return chain.stream(final_prompt, config={"run_name": "llm-stream"}), context_to_show
            else:
                logger.info("Detected non-Google model - using ChatPromptTemplate for multi-turn RAG.")

                message = [("system", system_prompt)] + conversation_history + [("user", "{question}")]
                
                self.print_conversation_history(message)

                prompt_template = ChatPromptTemplate.from_messages(message)
                chain = prompt_template | llm | StreamingFilterThinkParser | StrOutputParser()
                return chain.stream({"question": query, "context": docs}, config={"run_name": "llm-stream"}), context_to_show

        except ConnectTimeout as e:
            logger.warning("Connection timed out while making a request to the LLM endpoint: %s", e)
            return iter(["Connection timed out while making a request to the NIM endpoint. Verify if the NIM server is available."]), []

        except Exception as e:
            logger.warning("Failed to generate multi-turn response due to exception %s", e)
            print_exc()

            if "[403] Forbidden" in str(e) and "Invalid UAM response" in str(e):
                logger.warning("Authentication or permission error: Verify the validity and permissions of your NVIDIA API key.")
                return iter(["Authentication or permission error: Verify the validity and permissions of your NVIDIA API key."]), []
            elif "[404] Not Found" in str(e):
                logger.warning("Please verify the API endpoint and your payload. Ensure that the model name is valid.")
                return iter(["Please verify the API endpoint and your payload. Ensure that the model name is valid."]), []
            else:
                return iter([f"Failed to generate RAG chain with multi-turn response. {str(e)}"]), []


    def document_search(self, content: str, messages: List, reranker_top_k: int, vdb_top_k: int, collection_name: str = "", **kwargs) -> List[Dict[str, Any]]:
        """Search for the most relevant documents for the given search parameters.
        It's called when the `/search` API is invoked.

        Args:
            content (str): Query to be searched from vectorstore.
            num_docs (int): Number of similar docs to be retrieved from vectorstore.
            collection_name (str): Name of the collection to be searched from vectorstore.
        """

        logger.info("Searching relevant document for the query: %s", content)

        try:
            document_embedder = get_embedding_model(model=kwargs.get("embedding_model"), url=kwargs.get("embedding_endpoint"))
            vs = get_vectorstore(document_embedder, collection_name, kwargs.get("vdb_endpoint"))
            if vs is None:
                logger.error("Vector store not initialized properly. Please check if the vector db is up and running")
                raise ValueError()

            docs = []
            local_ranker = get_ranking_model(model=kwargs.get("reranker_model"), url=kwargs.get("reranker_endpoint"), top_n=reranker_top_k)
            top_k = vdb_top_k if local_ranker and kwargs.get("enable_reranker") else reranker_top_k
            logger.info("Setting top k as: %s.", top_k)
            retriever = vs.as_retriever(search_kwargs={"k": top_k})  # milvus does not support similarily threshold

            retriever_query = content
            if messages:
                if kwargs.get("enable_query_rewriting"):
                    # conversation is tuple so it should be multiple of two
                    # -1 is to keep last k conversation
                    history_count = int(os.environ.get("CONVERSATION_HISTORY", 15)) * 2 * -1
                    messages = messages[history_count:]
                    conversation_history = []

                    for message in messages:
                        if message.role !=  "system":
                            conversation_history.append((message.role, message.content))

                    # Based on conversation history recreate query for better document retrieval
                    contextualize_q_system_prompt = (
                        "Given a chat history and the latest user question "
                        "which might reference context in the chat history, "
                        "formulate a standalone question which can be understood "
                        "without the chat history. Do NOT answer the question, "
                        "just reformulate it if needed and otherwise return it as is."
                    )
                    query_rewriter_prompt = prompts.get("query_rewriter_prompt", contextualize_q_system_prompt)
                    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                        [("system", query_rewriter_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}"),]
                    )
                    q_prompt = contextualize_q_prompt | query_rewriter_llm | StreamingFilterThinkParser | StrOutputParser()
                    # query to be used for document retrieval
                    logger.info("Query rewriter prompt: %s", contextualize_q_prompt)
                    retriever_query = q_prompt.invoke({"input": content, "chat_history": conversation_history})
                    logger.info("Rewritten Query: %s %s", retriever_query, len(retriever_query))
                    if retriever_query.replace('"', "'") == "''" or len(retriever_query) == 0:
                        return []
                else:
                    # Use previous user queries and current query to form a single query for document retrieval
                    user_queries = [msg.content for msg in messages if msg.role == "user"]
                    retriever_query = ". ".join([*user_queries, content])
                    logger.info("Combined retriever query: %s", retriever_query)
            # Get relevant documents with optional reflection   
            if os.environ.get("ENABLE_REFLECTION", "false").lower() == "true":
                max_loops = int(os.environ.get("MAX_REFLECTION_LOOP", 3))
                reflection_counter = ReflectionCounter(max_loops)
                docs, is_relevant = check_context_relevance(content, retriever, local_ranker, reflection_counter, kwargs.get("enable_reranker"))
                if not is_relevant:
                    logger.warning("Could not find sufficiently relevant context after maximum attempts")
                return docs
            else:
                if local_ranker and kwargs.get("enable_reranker"):
                    logger.info(
                        "Narrowing the collection from %s results and further narrowing it to %s with the reranker for rag"
                        " chain.",
                        top_k,
                        reranker_top_k)
                    logger.info("Setting ranker top n as: %s.", reranker_top_k)
                    # Update number of document to be retriever by ranker
                    local_ranker.top_n = reranker_top_k

                    context_reranker = RunnableAssign({
                        "context":
                            lambda input: local_ranker.compress_documents(query=input['question'],
                                                                        documents=input['context'])
                    })

                    retriever = {"context": retriever, "question": RunnablePassthrough()} | context_reranker
                    docs = retriever.invoke(retriever_query, config={'run_name':'retriever'})
                    # Normalize scores to 0-1 range"
                    docs = normalize_relevance_scores(docs.get("context", []))
                    return docs
            docs = retriever.invoke(retriever_query, config={'run_name':'retriever'})
            # TODO: Check how to get the relevance score from milvus
            return docs

        except Exception as e:
            raise APIError(f"Failed to search documents. {str(e)}") from e

    def print_conversation_history(self, conversation_history: List[str] = None, query: str | None = None):
        if conversation_history is not None:
            for role, content in conversation_history:
                logger.info("Role: %s", role)
                logger.info("Content: %s\n", content)
        if query is not None:
            logger.info("Query: %s\n", query)
