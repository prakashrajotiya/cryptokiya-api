import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import ragRoutes from './routes/rag.js';

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api', ragRoutes);

app.get('/', (req, res) => {
  res.send('RAG API is running.');
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ error: err.message || 'Something went wrong!' });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
