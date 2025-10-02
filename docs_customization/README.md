# Notes about the customizing the NVIDIA RAG Pipeline

## Background

The [NVIDIA RAG Blueprint](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline) provides an useful template for creating enterprise-destined RAG pipelines. For most cases it is not ready to be used as such, but provides a set of useful building blocks that you may test out of the box. This blueprint is based on the NVIDIA NIM architecture and should be deployable anywhere where there is NVIDIA hardware and software stack available. For NVIDIA Enterprise AI and NIM-compatible infrastructure, please check the [documentation](https://docs.nvidia.com/ai-enterprise/release-7/7.1/support/support-matrix.html).

Our motivation to customize the bluepring to our needs was to add a user interface (we went with [Open WebUI](https://docs.openwebui.com]), ability to use also models not made by Meta and (so we added [Ollama](https://docs.ollama.com)) and experiment also with other document intelligence workflows, namely [Docling](https://github.com/docling-project/docling).

Also, we wanted the pipeline to run with 1-2 L40S GPUs, so optimizations and offloading tasks off GPU memory was needed.

When starting our work we forked the 2.0 version of the [NVDIA RAG Bluepring Github repo](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline). The repo has since been updated at least to version 2.2.1. Merging the changes to our current fork might be relevant in the future.

## Getting started

To run you need compatible hardware, a server with at least one L40S GPU is needed. For development and testing we used the services of [DataCruch](http://www.datacrunch.io).

We deployend a virtual Ubuntu Linux instance to DataCrunch and wrote our code with Visual Studio code Remote Server connection to the instance. This worked well. In this setup you deploy you instance preferably with generated SSH-keys and modify your ./ssh/config accordingly:

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

The container images should download and the containers start. After succesfull start you should open SSH tunnels from your local computer:

```bash
> ssh -i ~/.ssh/id_ed25519_datacrunch -L 3000:localhost:3000 -L 9091:localhost:9091 -L 5001:localhost:5001 root@<the ip/hosthame of your instance>
```

The Open WebUI user interface should now accessible at localhost:3000 with your web browser. At the first login, you need to create administrator user identity.

The system should now be ready to test. For the full capability some customizations are needed (see Configuration). When a conversation starts and a model is first selected, the first response is slow because the models need to get loaded to memory. The following responses are faster.

After use, you may stop the services by running the shutdown script at the remote server:

```bash
./deploy_rag_openwebui.sh
```

## Configuration

### Adding users, groups and giving access to models.

An Admin can access group and user management by clicking user menu lower left and selecting *Admin Panel*. There you click *Group* and create a group, for example *RAG users*. Now edit the group by clicking pen icon on the right at the user row and enable *Models Access*. Save settings. Now click *Overview* and and add an user with the *+* icon on the upper left. Type in the details and click *Save*. Now go back to *Groups*, select *RAG users* and add your created user to the group. 

Next, click again the user menu at lower left, click *Settings* and *Admin Settings*. Select *Models* and the model you want to give access to group *RAG users*. In model settings select the group and give *READ* permissions.

The user(s) belonging to *RAG users* should now be able to use the installed models.

### Setting the system parameters for document processing

Unfortunately not all parameters can be set using system variables and some need tweaking using the user interface.

#### Context extraction engine

Click user menu lower left, select *Settings* and *Admin Settings*. Change context extraction engine to *Docling* and check that the address of the engine is set to http://docling:5001. Put the address in the box if necessary. 

Next set Text splitter chuck size to 400 and chunck overlap to 100 if not set by default. This improves the robustness of information retrieval. Exceeding 400 might cause silent errors due to internal configuration of system.  

Check that *Embedding Model Engine* is set to *OpenAI* and the address is http://nim-proxy:8020/v1. *API Key* can be anything, *abcd* for example. *Embedding Model* should say nvidia/nv-embedqa-e5-v5.

Finally, set the reranking model. Enable *Hybrid Search*. Select *Reranking Engine* and choose *External*, for *Reranking Engine* type http://nim-proxy:8020/v1/ranking, for *API Key* type anything, *abcd*, for example and to *Reranking Model* type *nvidia/nv-rerankqa-mistral-4b-v3*.

Click *Save* to apply the changes and check for any error messages. The final settings should look like this:

**TODO:** Add up to date pic: ![Setup page for document content extraction]()

### Setting context length and token limits

TODO: Why setting a decent context window and token limit length is important.

## What is RAG and how to use it

TODO: Explain basic concepts and tips.

## About system variables

TODO: Write how to use .env and .env template

## Integrating Open WebUI

TODO: Explain

## Integrating Docling

TODO: Explain why Docling and why it is running CPU now. One should seriosly consider employing GPU if possible later.

## Using NIM proxy

TODO: Explain why we need nim-proxy - we need to make NIM services API look like OpenAI API. Maybe MCP could help in this?

## Future considerations

- Improving security. Root Signals, NVIDIA Garak, Snyk etc.
- Importing open source material input / scraping from internet.
- Decision whether to keep up with the blueprint development or diverge.
- MCP potential and security?
- Open WebUI enterprise option with hardening support etc.
- Long term visions with scalability. Needs Kubernetes platform and orchestration?

## Cleanup

TODO: Thoughts about cleanup here.

```bash
docker compose -f deploy/compose/cleanup.yaml up -d --build
```

## Misc

Docling Serve web UI is available at http://localhost:5001/ui/
Milvus management interface is available at http://localhost:9091/webui/

## TODO

- https reverse proxy implementation
- Automatic cleanup / trash collection, to run every night midnight forced and perhaps more frequenty sensing whic users are active and only cleaning closed sessions. Logout trigger might be possible as well
- Milvus cluster ETCD metastore at http://127.0.0.1:2379 reports Unhealthy status
- Change all log levels to minimum, maybe "ERROR" or "WARNING"/"WARN"
- Deploy milvus to GPU if better GPU (more memory) is available
- Deploy Docling to GPU if better GPU (more memory)
