import express from 'express';
import multer from 'multer';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { createClient } from '@supabase/supabase-js';
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Document } from '@langchain/core/documents';
import dotenv from 'dotenv';

dotenv.config();

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });

// Initialize Supabase Client
const supabaseUrl = process.env.SUPABASE_URL || '';
const supabaseKey = process.env.SUPABASE_PRIVATE_KEY || '';
const supabaseClient = createClient(supabaseUrl, supabaseKey);

// Initialize Gemini Embeddings and Chat Model
// Note: GOOGLE_API_KEY must be in the .env file.
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-2",
  modelName: "gemini-embedding-2",
});

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-flash-lite-latest",
  modelName: "gemini-flash-lite-latest",
  temperature: 0,
});

/**
 * @route POST /api/ingest
 * @desc Uploads a PDF, chunks it, embeds it, and stores in Supabase pgvector
 */
router.post('/ingest', upload.single('file'), async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded. Please upload a PDF file using the "file" field.' });
    }

    if (req.file.mimetype !== 'application/pdf') {
      return res.status(400).json({ error: 'Only PDF files are supported.' });
    }

    console.log('Sending PDF to Gemini for OCR extraction...');

    // Convert buffer to base64 for Gemini
    const base64Pdf = req.file.buffer.toString("base64");

    // Initialize standard Gemini client (not LangChain wrapper) for File API
    const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
    const model = genAI.getGenerativeModel({ model: "gemini-3.5-flash" });

    const result = await model.generateContent([
      "Extract all text from this document verbatim. Do not summarize or add formatting. Just give me the raw text found in the document.",
      {
        inlineData: {
          data: base64Pdf,
          mimeType: "application/pdf"
        }
      }
    ]);

    const extractedText = result.response.text();

    if (!extractedText || extractedText.trim().length === 0) {
      return res.status(400).json({ error: 'Could not extract text from the PDF even with Gemini OCR.' });
    }

    console.log('Splitting extracted text into chunks...');
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await splitter.createDocuments([extractedText], [{ source: req.file.originalname }]);

    console.log(`Generated ${docs.length} chunks. Storing in Supabase...`);

    // Store in Supabase using Langchain's integration
    await SupabaseVectorStore.fromDocuments(
      docs,
      embeddings,
      {
        client: supabaseClient,
        tableName: "documents",
        queryName: "match_documents",
      }
    );

    console.log('Ingestion complete.');
    res.json({ message: 'File successfully ingested and vectorized.', chunks: docs.length });
  } catch (error) {
    next(error);
  }
});

/**
 * @route POST /api/chat
 * @desc Asks a question, retrieves context from pgvector, and generates answer via Gemini
 */
router.post('/chat', async (req, res, next) => {
  try {
    const { question } = req.body;

    if (!question) {
      return res.status(400).json({ error: 'Question is required in the request body.' });
    }

    // Initialize VectorStore from existing data
    const vectorStore = new SupabaseVectorStore(embeddings, {
      client: supabaseClient,
      tableName: "documents",
      queryName: "match_documents",
    });

    const retriever = vectorStore.asRetriever({
      k: 4, // Retrieve top 4 most relevant chunks
    });

    // Create prompt template
    const prompt = ChatPromptTemplate.fromTemplate(`
      You are an AI assistant designed to answer questions based on the provided context.
      If the answer is not in the context, clearly state that you do not know. Do not guess.

      Context: {context}

      Question: {input}

      Answer:
    `);

    console.log(`Answering question: "${question}"...`);

    // Retrieve documents
    const retrievedDocs = await retriever.invoke(question);
    const contextText = retrievedDocs.map(doc => doc.pageContent).join("\\n\\n");

    // Format prompt and invoke model
    const promptValue = await prompt.invoke({ context: contextText, input: question });
    const response = await llm.invoke(promptValue);

    res.json({
      answer: response.content,
      context: retrievedDocs.map(doc => doc.pageContent),
    });
  } catch (error) {
    next(error);
  }
});

/**
 * @route DELETE /api/clear
 * @desc Deletes all documents from the Supabase pgvector store
 */
router.delete('/clear', async (req, res, next) => {
  try {
    console.log('Clearing all documents from the knowledge base...');

    // Delete all records from the documents table
    // Supabase requires a filter for deletes, so we use .not('id', 'is', null) to match all rows
    const { error } = await supabaseClient
      .from('documents')
      .delete()
      .not('id', 'is', null);

    if (error) {
      throw error;
    }

    console.log('Knowledge base cleared successfully.');
    res.json({ message: 'All documents have been successfully removed from the knowledge base.' });
  } catch (error) {
    next(error);
  }
});

export default router;
