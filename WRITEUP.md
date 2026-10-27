Making PDF ingestion less painful: a hands-on OCR comparison

Data ingestion from PDFs, images, complex documents and HTML has always been a hard step. Accurately recognizing handwriting, LaTeX formulas and tables directly affects training quality, RAG knowledge bases and ontology stacks.

I researched the popular open source PDF ingestion frameworks and built a demo so you can see the differences yourself: [https://ocr-playground.srv1567269.hstgr.cloud/]

What’s in the demo

A pre-run comparison on a complex 6-page PDF mixing handwriting, tables and diagrams
A live upload feature with two paths:
Fast path: LiteParse + Tesseract on CPU
Slow path: VLM-based OCR models served on Modal’s serverless GPUs

The GPU path is far more accurate but suffers from cold starts. To reduce that, I configured Modal to cache two things on a volume:

Model weights downloaded from Hugging Face (saves minutes of network transfer)
torch.compile artifacts for vLLM (saves about 27s)

Cold start dropped to roughly 2-3 minutes, mostly container start cost, plus a 2-minute warm window afterward.

What I observed (a qualitative comparison on this one document, not a formal benchmark)

On PDFs with images, handwriting or complex layouts, GPU-based OCR is clearly better than the CPU path.
Top tier: Chandra OCR 2 and Nemotron 3.5 Parse. Both make small mistakes on pages 5 and 6, but overall quality is excellent. Chandra OCR 2 produced the cleanest table layout. Nemotron 3.5 Parse kept very good quality and was noticeably faster than the others, so the trade-off is layout clarity vs. speed.
Second tier: GLM-OCR and MinerU 2.5.

What I’m exploring next

OCR for web pages
Other inference serving options: RunPod (containers and serverless), or self-hosting with llm-d + KEDA + vLLM
Agentic search over the knowledge base: OCR chunks + embeddings + metadata indexed in Elasticsearch, using hybrid retrieval (BM25 + kNN merged with Reciprocal Rank Fusion) rather than pure vector search. At enterprise scale, NVIDIA cuVS can build HNSW graphs on the GPU, speeding up ingest and reindexing of large vector datasets (check version support): https://www.elastic.co/docs/solutions/search/vector/gpu-vector-indexing

Would love to hear which models you’d add to the comparison.

#OCR #RAG #DocumentAI #LLM #Elasticsearch