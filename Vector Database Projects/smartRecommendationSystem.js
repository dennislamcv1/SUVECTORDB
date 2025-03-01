const fs = require("fs");
const pdf = require("pdf-parse");
const { HfInference } = require("@huggingface/inference");
const readline = require("readline");
const { ChromaClient } = require("chromadb");

const hf = new HfInference("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");

const chroma = new ChromaClient();
const collectionName = "job_postings";

const smartRecommendationSystem = require('./jobPostings.js');

// Function to extract text from a PDF file
const extractTextFromPDF = async (filePath) => {
    try {
      const dataBuffer = fs.readFileSync(filePath);
      const data = await pdf(dataBuffer);
      const text = data.text.replace(/\n/g, " ").replace(/ +/g, " ");
      return text;
    } catch (err) {
      console.error("Error extracting text from PDF:", err);
      throw err;
    }
  };
  // Function to convert text to embeddings using Hugging Face embeddings
  const generateEmbeddings = async (text) => {
      try {
        const result = await hf.featureExtraction({
          model: "sentence-transformers/all-MiniLM-L6-v2",
          inputs: text,
        });
        // console.log("Embedding API result:", result); // Log the entire result
    return result
      } catch (err) {
        console.error("Error converting text to embeddings:", err);
        throw err;
      }
    };
    // Function to read user input
  const promptUserInput = (query) => {
      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
      });
      return new Promise((resolve) =>
        rl.question(query, (answer) => {
          rl.close();
          resolve(answer);
        })
      );
    };
  // Function to  extract the text data from a PDF resume file asynchronously.
  const extractJobs = (text) => {
    const jobsPattern = /Jobs([\s\S]+?)For /i;
    const jobsMatch = jobsPattern.exec(text);
    if (jobsMatch) {
      return jobsMatch[1].split(/[^a-zA-Z0-9]+/).map(item => item.trim().toLowerCase()).filter(item => item.length > 0);
    }
    return [];
  };
    
  // Function to store embeddings in Chroma DB
  const storeEmbeddingsInChromaDB = async (jobPostings) => {
    const jobEmbeddings = [];
    const metadatas = jobPostings.map(() => ({})); // Empty metadata objects
    for (const job of jobPostings) {
      const embedding = await generateEmbeddings(job.jobDescription.toLowerCase());
       jobEmbeddings.push(embedding);
    }
    const ids = jobPostings.map((_, index) => index.toString());
    const jobTexts = jobPostings.map(job => job.jobTitle);
  
    try {
      const collection = await chroma.getOrCreateCollection({ name: collectionName });
      
      await collection.add({
        ids: ids,
        documents: jobTexts,
        embeddings: jobEmbeddings,
        metadatas: metadatas,
      });
      console.log("Stored embeddings in Chroma DB.");
    } catch (error) {
      console.error("Error storing embeddings in Chroma DB:", error);
      throw error;
    }
  };
  // Main function to run the extraction, embedding generation and job recommendation process
  const main = async () => {
      try {
        await storeEmbeddingsInChromaDB(smartRecommendationSystem);
    
        // Extract and process the recipe PDF
        const filePath = await promptUserInput("Enter the path to the resume PDF: ");
        const text = await extractTextFromPDF(filePath);
        const jobs = extractJobs(text);
    
        if (jobs.length > 0) {
          console.log("Extracted Jobs:", jobs);
    
          // Generate embedding for the extracted ingredients
          const resumeEmbedding = await generateEmbeddings(jobs.join(' ').toLowerCase());
    
          // Query Chroma DB for similar recipes
          const collection = await chroma.getCollection({ name: collectionName });
          const results = await collection.query({
            queryEmbeddings: [jobEmbedding],
            n: 5, // Get top 5 similar items
          });
    
          console.log("Chroma DB Query Results:", results);
    
          if (results.ids.length > 0 && results.ids[0].length > 0) {
            console.log("Recommended Jobs:");
            results.ids[0].forEach((id, index) => {
              const recommendedItem = jobItems[parseInt(id)];
              console.log(`Top ${index + 1} Recommended Item ==> ${recommendedItem.job_name}`);
            });
          } else {
            console.log("No similar jobs found.");
          }
        } else {
          console.log("No jobs found.");
        }
      } catch (err) {
        console.error("An error occurred:", err);
      }
    };
    
  // Run the main function
  main();