const { ChromaClient } = require("chromadb");
const client = new ChromaClient();
const { HfInference } = require("@huggingface/inference");
const hf = new HfInference("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");
const collectionName = "job_collection";

async function generateEmbeddings(texts) {
  const results = await hf.featureExtraction({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    inputs: texts,
  });
  return results;
}

async function classifyText(word, labels) {
  const response = await hf.request({
    model: "facebook/bart-large-mnli",
    inputs: word,
  });

  if (!response || !response.labels || !response.scores) {
    console.error("Invalid response from classification model");
    return { labels: [], scores: [] };
  }

  return response;
}

async function extractFilterCriteria(query) {
  const criteria = { location: null, jobTitle: null, company: null, jobType: null };
  const labels = ["location", "job title", "company", "job type"];
 
  const words = query.split(" ");
  for (const word of words) {
    const result = await classifyText(word, labels);
    console.log('result', result);
    const highestScoreLabel = result.labels[0];
    const score = result.scores[0];
 
    if (score > 0.5) {
      switch (highestScoreLabel) {
        case "location":
          criteria.location = word;
          break;
        case "job title":
          criteria.jobTitle = word;
          break;
        case "company":
          criteria.company = word;
          break;
        case "job type":
          criteria.jobType = word;
          break;
        default:
          break;
      }
    }
  }
  console.log('Extracted Filter Criteria:', criteria);
  return criteria;
}

async function performSimilaritySearch(collection, queryTerm, jobPostings) {
  try {
    const queryEmbedding = await generateEmbeddings([queryTerm]);
    const results = await collection.query({
      collection: collectionName,
      queryEmbeddings: queryEmbedding,
      n: 3,
    });

    if (!results || results.length === 0) {
      console.log(`No similar results found to "${queryTerm}"`);
      return [];
    }

    let topJobPostings = results.ids[0].map((id, index) => {
      return {
        id,
        score: results.distances[0][index],
        jobTitle: jobPostings.find(item => item.job_id.toString() === id).jobTitle,
        jobType: jobPostings.find(item => item.job_id.toString() === id).jobType,
        jobDescription: jobPostings.find(item => item.job_id.toString() === id).jobDescription,
        company: jobPostings.find(item => item.job_id.toString() === id).company
      };
    }).filter(Boolean);
    return topJobPostings.sort((a, b) => b.score - a.score);
  } catch (error) {
    console.error("Error during similarity search:", error);
    return [];
  }
}

async function main() {
  const query = "Creative Studio";
  try {
    const collection = await client.getOrCreateCollection({ name: collectionName });
    
    const uniqueIds = new Set();
    jobPostings.forEach((job, index) => {
      while (uniqueIds.has(job.job.toString())) {
        job.job = `${job.job}_${index}`;
      }
      uniqueIds.add(job.job.toString());
    });

    const jobTexts = jobPostings.map((job) => `${job.jobTitle}. ${job.jobDescription}. ${job.jobType}. ${job.location}.`);
    const embeddingsData = await generateEmbeddings(jobTexts);
    
    const metadatas = jobPostings.map((job) => ({
      location: job.location,
      jobTitle: job.jobTitle,
      company: job.company,
    }));

    await collection.add({
      ids: jobPostings.map((job) => job.jobTitle.toString()),
      documents: jobTexts,
      embeddings: embeddingsData,
      metadatas: metadatas,
    });

    const filterCriteria = await extractFilterCriteria(query);
    console.log('Filter Criteria:', filterCriteria);
    
    const initialResults = await performSimilaritySearch(collection, query, jobPostings);
    initialResults.slice(0, 3).forEach((item, index) => {
      console.log(`Top ${index + 1} Recommended Job Title: ${item.jobTitle}`);
      console.log(`Top ${index + 1} Recommended Job Type: ${item.jobType}`);
      console.log(`Top ${index + 1} Recommended Job Description: ${item.jobDescription}`);
      console.log(`Top ${index + 1} Recommended Company: ${item.company}`);
    });
  } catch (error) {
    console.error("Error:", error);
  }
}

// Run the main function
main();