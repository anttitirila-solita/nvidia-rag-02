# Notes about the customizing the NVIDIA RAG Pipeline for internal use

## Important notes before start

- **The system must be deployed in an internet-accessible environment for the initial startup.** After the first successful launch, it can be moved to an internal network.

## Background

The [NVIDIA RAG Blueprint](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline) provides useful template for creating enterprise-destined RAG pipelines. For most cases it is not production-ready as-is, but provides a set of useful building blocks that you may test out of the box. This blueprint is based on the NVIDIA NIM architecture. It should be deployable anywhere NVIDIA hardware and the NIM software stack are available.. For NVIDIA Enterprise AI and NIM-compatible infrastructure, please check the [documentation](https://docs.nvidia.com/ai-enterprise/release-7/7.1/support/support-matrix.html).

Our motivation to customize the blueprint to our needs was to add a user interface (we went with [Open WebUI](https://docs.openwebui.com]), support for models beyond those provided by Meta and (so we added [Ollama](https://docs.ollama.com)) and experiment also with other document intelligence workflows, namely [Docling](https://github.com/docling-project/docling). There is also possibility to use [vLLM](https://docs.vllm.ai) to run models more efficiently. vLLM is commented out in our current deployment. Instructions on how to enable it are in chapter *Enabling vLLM* below.

We also aimed to run the pipeline on one or two L40S GPUs, requiring memory optimizations and selective offloading from GPU memory.

When starting our work we forked the 2.0 version of the [NVIDIA RAG Blueprint GitHub repository](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline). The repo has since been updated at least to version 2.2.1. Merging the changes to our current fork might be relevant in the future.

## Prerequisites

To run this pilot we need:

- At least one NVIDIA L40S GPU and a physical or virtual server
- NVIDIA AI Enterprise license or developer credentials to run NVIDIA NIM containers

## Getting started

A compatible server with at least one L40S GPU is required. For development and testing we used the services of [DataCrunch](http://www.datacrunch.io). NVIDIAs cloud service [Brev](https://brev.nvidia.com/) is an easy option, although it is a bit on the expensive side and does not offer customization options for the runtime environment etc. as DataCruch for example does.

We deployend a virtual Ubuntu Linux instance to DataCrunch and wrote our code with Visual Studio code Remote Server connection to the instance. This worked well. Deploy your instance preferably using generated SSH keys and configure your ~/.ssh/config file as follows:

```bash
Host <the ip/hosthame of your instance>
        HostName <the ip/hosthame of your instance>
        IdentityFile=<your home dir>/.ssh/id_ed25519_datacrunch
        User root
```

After this you may connect to the ip/hostname of instance with Visual Studio code. Open the the Visual Studio code Terminal or an SSH terminal and start the containers with:

```bash
./deploy_rag_openwebui.sh
```

The container images should download and the containers start. After a successful start, open SSH tunnels from your local machine:

```bash
> ssh -i ~/.ssh/id_ed25519_datacrunch -L 3000:localhost:3000 -L 9091:localhost:9091 -L 5001:localhost:5001 root@<the ip/hosthame of your instance>
```

The Open WebUI interface should now be accessible at *http://localhost:3000*. On the first login, create an administrator account.

The system should now be ready to test. For the full capability some customizations are needed (see Configuration). When a conversation starts and a model is first selected, the first response is slow because the models need to get loaded to memory. The following responses are faster.

After use, you may stop the services by running the shutdown script at the remote server:

```bash
./deploy_rag_openwebui.sh
```

You may monitor the system usage, especially RAM and GPU memory consumption with *htop* and *nvidia-smi*.

## Configuration

### Adding users, groups and giving access to models

An Admin can access group and user management by clicking user menu lower left and selecting *Admin Panel*. There you click *Group* and create a group, for example *RAG users*. Now edit the group by clicking pen icon on the right at the user row and enable *Models Access*. Save settings. Now click *Overview* and add a user with the *+* icon on the upper left. Type in the details and click *Save*. Now go back to *Groups*, select *RAG users* and add your created user to the group.

Next, click again the user menu at lower left, click *Settings* and *Admin Settings*. Select *Models* and the model you want to give access to group *RAG users*. In model settings select the group and give *READ* permissions.

The user(s) belonging to *RAG users* should now be able to use the installed models.

### Setting the system parameters for document processing

Unfortunately not all parameters can be set using system variables and some need tweaking using the user interface.

#### Context extraction engine

Click user menu lower left, select *Settings* and *Admin Settings*. Change context extraction engine to *Docling* and check that the address of the engine is set to *http://docling:5001*. Put the address in the box if necessary.

Next set Text splitter chuck size to 400 and chunk overlap to 100 if not set by default. This improves the robustness of information retrieval. Exceeding a chunk size of 400 may cause silent errors due to internal configuration constraints.

Check that *Embedding Model Engine* is set to *OpenAI* and the address is *http://nim-proxy:8020/v1*. *API Key* can be anything, *abcd* for example. *Embedding Model* should say *nvidia/nv-embedqa-e5-v5*.

Finally, set the reranking model. Enable *Hybrid Search*. Select *Reranking Engine* and choose *External*, for *Reranking Engine* type *http://nim-proxy:8020/v1/ranking*, for *API Key* type anything, *abcd*, for example and to *Reranking Model* type *nvidia/nv-rerankqa-mistral-4b-v3*.

Click *Save* to apply the changes and check for any error messages. The final settings should look like this:

[Setup for document content extraction](./pics/doc_processing_settings.png)

### Setting context length and token limits

Context length determines how much conversation history the model retains. While larger context windows provide more memory, they also increase GPU memory consumption and may reduce model accuracy beyond a certain point. Every model has a maximum context length, although using the max length is not always optimal. The model usually starts forgetting things when using excessively long context lengths. Context length also consumes GPU memory.

In our tests we used context length of 8192. With bigger and more capable GPU (upgrade from NVIDIA L40S) the context length can be longer.

Use chat settings on the respective discussion to increase model context length. Context length is the amount of tokens given to the Ollama model used. Max Tokens is the maximum length of the answer.

[Setup for token lengths](./pics/chat_token_length.png)

## What is RAG and how to use it in this pilot

### RAG background

RAG (Retrieval Augmented Generation) allows querying and reasoning over data not included in the original training corpus of the language model. Typically RAG is used so that the user drops their on set of documents to the chat. This is our supported use case. More advanced use cases could include importing enterprise data to a local database in the RAG system for general use or role-based access.

### The current implementation

We use the NVIDIA RAG blueprint as the baseline. We have modified the pipeline and choose not to deploy many components of the implementation such as Guardrails or LLM-powered review or refinement of answers due to L40S memory constraints. Also while NVIDIA releases high-quality NIMs, the NVIDIA NIM model catalog is limited, and new models are released less frequently compared to platforms like Ollama. The NVIDIA blueprint also lacks good enterprise-level UI. We chose to implement Open WebUI while trying to keep the changes to the base Open WebUI at minimum.

Due to the current limitations with [NVIDIA RAG Blueprint] components we decided to rely mostly on the [Open WebUI RAG](https://docs.openwebui.com/tutorials/tips/rag-tutorial) functionality. Instructions on setting the [RAG pipeline](https://docs.openwebui.com/features/rag#enhanced-rag-pipeline) are relevant reading. In short, we use Open WebUI RAG pipeline but use NVIDIA NIMs where they make the most sense, such as vector database Milvus engine, embedding engine, reranking etc. To integrate these to the Open WebUI proxy.

### About the terms of use

For this use case we used the abovementioned NVIDIA Blueprint to build an enterprise RAG pipeline. However, many language-model components or the pipeline could be under Meta LLama Community License Agreement (various versions) that limits their use in certain purposes [Llama 3.3. acceptable use](https://www.llama.com/llama3_3/use-policy/). NVIDIA Llama derived models however do not necessary refer to Llama Community License Agreement. You may check this at [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/). For example, some Nemotron models do not refer to the Llama Community License, Llama-3.1-Nemotron-Nano-8B-v1 for example, and some including Llama-3.3-nemotron-super-49b-v1.5 do. [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and other NVIDIA agreements regarding use.

## About system variables and .env

The application and underlying microarchitecture take a fair bit of settings. We have collected them to .env.template. For the deployment one should fill the missing bits with values that the deployment needs and save the result as .env-file. **One should not push the .env file to a repository.** .env file has potentially confidential information including API keys.

Most importantly one needs NVIDIA Enterprise AI API key to pull and run the needed NIM containers. As a developer, one can register to [https://developer.nvidia.com/](https://developer.nvidia.com/) and gain access to at least some of the needed containers and resources. For production one should have a proper NVIDIA Enterprise AI license and appropriate keys.

In short, the .env file specifies model versions, service addresses, and runtime parameters such as timeout values.

The current environments are mostly collected to the .env(.template), but some are still hardcoded to deployment file (docker-compose) in *./deploy/compose*. This is true especially with *./deploy/compose/openwebui.yaml* that controls the Open WebUI setup.

## Integrating Open WebUI

Open WebUI works as standalone UI. So far we have not made any code changes to the native release. In Open WebUI we have configured which services to use including context extraction engine components and the base Ollama hosted language models.

Open WebUI uses OpenAI APIs to interact with other services. These are not entirely compatible with NVIDIA NIM APIs. This is why we have incorporated a simple NIM proxy in *./nim_proxy* to modify the API calls. This proxy can be expanded if needed. For cleaner setup a tech such as MCP could be incorporated once the support in Open WebUI and NVIDIA Enterprise AI perhaps matures a bit.

Open WebUI also offers a licensed enterprise version with potential to hardening and customization.

## Integrating Docling

[Docling](https://github.com/docling-project/docling) is an IBM-driven OSS initiative for better document understanding. In this implementation docling bypasses NVIDIA’s built-in document intelligence components such as NeMo Retrievers and Paddle OCR. This decision was made firstly because Docling is easy to deploy stand-alone and then integrate that to Open WebUI RAG functionality. Secondly the NVIDIA document intelligence currently uses language models with licensing restriction. As such the NVIDIA document intelligence appears capable and fast. Should it become more easy to integrate and license concerns potentially easier, it could be considered as an option in the future.

Currently, Docling’s CPU implementation is single-threaded and relatively slow. The DOCLING_SERVE_MAX_SYNC_WAIT=600 limit is often exceeded without propagating errors back to Open WebUI, resulting in apparent infinite loops. For user this shows as an eternal processing loop that need to be cancelled manually. It should be considered if there is GPU memory budget to move to GPU-implementation of Docling. There might still be optimizations not fully explored as well. Currently Docling performance is borderline defective.

Docling pipeline can be tested with its standalone UI in [http://localhost:5001/ui/](http://localhost:5001/ui/). This is informative for example when tweaking settings for Open WebUI.

So far we have not discovered potential options of document image intelligence options beyond OCR. These include keyword generation and graph extraction or conversion to Mermaid format. These capabilities are developing and one could keep an eye on what is being integrated to both Docling and OpenWebUI.

## Evaluations

It is important to collect feedback to feed the future development of the application and understand user needs. Open WebUI has mechanism for this. Unfortunately by default exposes also the content of the discussion of which the feedback was given. It is undecided how to prevent the chats from being linked to feedback or if there is an option to just leave positive or negative score along with free text. Also one could test if cleaning up the chat database (section "Cleanup" below) sanitizes the feedback from chat contents as well.

## Qualitative testing of RAG

A good RAG system should be effective in retrieving relevant information and forming informative answers based on the information. There are tools such as [Phoenix](https://phoenix.arize.com/) for this. A simple approach is to run a series of documents with predefined questions and manually rank the results. Alternatively the LLM-as-a-judge approach could be used. One could develop a repeatable test pattern and re-run that when testing new configurations for deployment. A simple example of implementation is available in *./testing*.

## Docling performance testing

TODO: Explain how to performance test Docling APIs and potentially other functionality.

## Cleanup

Open WebUI leaves dropped documents in as temporary files in the server. Milvus vector database creates a collection for every document that is left there after discussion. Chats are archived in Open WebUI SQLite database. We have scripted an approach to delete these:

```bash
docker compose -f deploy/compose/cleanup.yaml up -d --build
```

The cleanup script drops tables from the [Open WebUI Internal SQLite Database](https://docs.openwebui.com/tutorials/tips/sqlite-database#user-table).


Once Open WebUI evolves there might be other locations were chat or document information is left over. One should keep an eye for these potential leak points.

## Enabling vLLM

vLLM is an optimized model serving library that can improve performance and memory usage when running large language models. In our deployment vLLM is included but commented out. To enable it, follow these steps:
Uncomment the vLLM service in the *deploy/compose/openwebui.yaml* file by removing the `#` characters before the lines ENABLE_OPENAI_API and OPENAI_API_BASE_URL environment variables. Some models need HUGGING_FACE_HUB_TOKEN environment variable in *.env* file.

Uncomment the vLLM service deploy also in deploy_rag_openwebui.sh script and shutdown_rag_openwebui.sh script. There is also a separate deploy_vLLM.sh script that can be used to deploy only vLLM service if needed.

Check from Open WebUI admin settings that in *Settings -> Admin Settings -> Connections* the OPENAI API is enabled and Manage OpenAI API Connections is set to http://vllm:8080/v1.

If you use both ollama and vLLM, memory usage can be an issue if you are using non- quantized models. Consider using quantized models in vLLM to reduce memory consumption.
For example gemma-3-12b model takes 57GB of GPU memory in non-quantized form if run in vLLM.

## Troubleshooting and potential pain points

- RAG may not always see the offered content. [Open WebUI Troubleshooting RAG](https://docs.openwebui.com/troubleshooting/rag/) might have suggestions.
- Docling performance and DOCLING_SERVE_MAX_SYNC_WAIT=600. The Open WebUI does not expose the status nor recovery properly. 
- Capability of the OCR pipeline in detecting text from images. The reliability of this has not been thoroughly tested.

## Future considerations

- Improving security. Root Signals, NVIDIA Garak, Snyk etc.
- Importing open source material input / scraping from internet. FireCrawl and Crawl4AI are interesting options.
- Decision whether to keep up with the blueprint development or diverge to own path
- [MCP](https://docs.openwebui.com/openapi-servers/mcp) potential and security?
- Open WebUI enterprise option with hardening support etc.
- Long term visions with scalability. Needs Kubernetes platform and orchestration?
- How to keep up and deploy fresh Open WebUI versions?
- There are small and capable models such as Qwen. Exploring them could make sense at some point.
- Agentic workflows are taking over the workflows. [NVIDIA Safety for Agentic AI](https://build.nvidia.com/nvidia/safety-for-agentic-ai), [NVIDIA Build an AI Agent for Enterprise Research](https://build.nvidia.com/nvidia/aiq)
- How to collect and analyze feedback safely without exposing the discussions of the users. This critical for further development
- What is the delivery model, acceptable level of reliability 
- Integrate to organization HTTPS proxy
- Use organization authentication, OAUTH, and especially [Azure AD Domain Services LDAPS](https://docs.openwebui.com/tutorials/offline-mode)
- Cross-check against the [Open WebUI offline mode instructions](https://docs.openwebui.com/tutorials/offline-mode)
- RAG system prompt update and enhancement. Which language(s) should one use, for example?


## Misc

Docling Serve web UI is available at http://localhost:5001/ui/
Milvus management interface is available at http://localhost:9091/webui/

## TODO

- https reverse proxy implementation
- Automatic cleanup / trash collection, to run every night midnight forced and perhaps more frequently sensing which users are active and only cleaning closed sessions. Logout trigger might be possible as well
- Milvus cluster ETCD metastore at http://127.0.0.1:2379 reports Unhealthy status
- Change all log levels to minimum, maybe "ERROR" or "WARNING"/"WARN"
- Deploy milvus to GPU if better GPU (more memory) is available
- Deploy Docling to GPU if better GPU (more memory). Maybe switch to H100 and offload Docling to GPUs.
