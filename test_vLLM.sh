curl -X POST "http://localhost:8001/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "google/gemma-3-12b-it",
		"messages": [
			{
				"role": "user",
				"content": "What is the capital of France?"
			}
		]
	}'