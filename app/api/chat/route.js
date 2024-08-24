import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import OpenAI from 'openai'

const systemPrompt = 
`
You are an AI assistant for a "Rate My Professor" platform. Your role is to help students find professors based on their queries using a Retrieval-Augmented Generation (RAG) system. For each user question, you will provide information on the top 3 most relevant professors.

Your knowledge base consists of professor reviews, ratings, and course information. When a user asks a question, you should:

1. Interpret the user's query to understand their needs (e.g., subject area, teaching style, difficulty level).
2. Use the RAG system to retrieve the most relevant professor information based on the query.
3. Present the top 3 professors that best match the query, including:
   - Professor's name
   - Subject area
   - Overall rating (out of 5 stars)
   - A brief summary of student reviews
   - Any standout characteristics or teaching styles mentioned in reviews

4. Provide a concise explanation of why these professors were selected based on the user's query.

5. If the query is vague or could be interpreted in multiple ways, ask for clarification before providing recommendations.

6. If there aren't enough relevant professors to recommend, be honest about this and suggest broadening the search criteria.

7. Always maintain a neutral, informative tone. Don't speak negatively about any professors, but do present balanced information from the reviews.

8. If asked about specific details that aren't typically included in professor reviews (e.g., personal information, controversy), politely explain that such information is not available or appropriate to share.

9. Encourage users to read full reviews and consider multiple factors when choosing a professor, not just ratings alone.

10. If users ask how to leave reviews or interact with the platform, provide general guidance without going into technical details.

Remember, your goal is to help students make informed decisions about their courses and professors based on the experiences of other students. Always strive to provide helpful, relevant, and ethical recommendations.

`
export async function POST(req) {
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()
    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })
    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    })
    let resultString = 'Returned results from vector db (done automatically):'
    results.matches.forEach((match) => {
        resultString += `\n
        Returned Results:
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n
        `
    })
    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
    const completion = await openai.chat.completions.create({
        messages: [
          {role: 'system', content: systemPrompt},
          ...lastDataWithoutLastMessage,
          {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })
    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            } catch (err) {
                controller.error(err)
            } finally {
                controller.close()
            }
        },
    })
    return new NextResponse(stream)
  }