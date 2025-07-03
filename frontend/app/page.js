'use client';
import { useState } from 'react';

const questions = [
  "How have you been feeling lately?",
  "Do you feel motivated to do your daily tasks? Why or why not?",
  "Have you been sleeping and eating normally? Any changes?",
  "Do you find it easy or hard to focus or concentrate?",
  "Do you talk to friends/family regularly, or have you been avoiding social contact?",
  "Is there anything recently that's been stressing you out or making you anxious?",
  "Do you often feel sad, hopeless, or emotionally numb?",
  "Is there anything you want to change about how you’re currently feeling or coping?"
];

export default function Home() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState(Array(questions.length).fill(''));
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const updated = [...answers];
    updated[currentIndex] = e.target.value;
    setAnswers(updated);
  };

  const handleSubmit = async () => {
    setLoading(true);
    const transcript = questions.map((q, i) => `${q} ${answers[i]}`).join(' ');
    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript }),
      });
      const data = await res.json();
      setResult(data);
    } catch {
      setResult({ error: 'Something went wrong.' });
    }
    setLoading(false);
  };

  const isLast = currentIndex === questions.length - 1;

  return (
    <main className="min-h-screen flex items-start justify-center bg-zinc-900 text-white px-6 py-16">
      <div className="w-full max-w-2xl bg-zinc-800 rounded-2xl shadow-lg p-8 space-y-6">
        <h1 className="text-3xl font-bold text-center mb-2">Mental Health Check</h1>
        <p className="text-center text-gray-400 mb-4">Question {currentIndex + 1} of {questions.length}</p>

        <div>
          <label className="block font-semibold mb-2">{questions[currentIndex]}</label>
          <textarea
            className="w-full p-4 rounded-lg bg-zinc-700 text-white border border-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-400"
            rows={4}
            value={answers[currentIndex]}
            onChange={handleChange}
            placeholder="Type your answer here..."
          />
        </div>

        <div className="flex justify-between items-center pt-4">
          <button
            onClick={() => setCurrentIndex((i) => Math.max(i - 1, 0))}
            disabled={currentIndex === 0}
            className="px-4 py-2 rounded bg-gray-600 hover:bg-gray-700 disabled:opacity-40"
          >
            Back
          </button>

          {isLast ? (
            <button
              onClick={handleSubmit}
              className="px-6 py-2 rounded bg-blue-600 hover:bg-blue-700 font-semibold"
            >
              {loading ? 'Analyzing...' : 'Submit'}
            </button>
          ) : (
            <button
              onClick={() => setCurrentIndex((i) => Math.min(i + 1, questions.length - 1))}
              disabled={!answers[currentIndex].trim()}
              className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-700 disabled:opacity-40"
            >
              Next
            </button>
          )}
        </div>

        {result && (
          <div className="bg-white text-black rounded-xl p-5 mt-6 shadow-lg">
            <h2 className="text-xl font-bold mb-2">Result</h2>
            <p>
              <strong>Depression:</strong>{' '}
              {result.depression_predicted === 1 ? (
                <span className="text-red-600 font-semibold">Positive</span>
              ) : (
                <span className="text-green-600 font-semibold">Negative</span>
              )}
            </p>
            <p>
              <strong>Anxiety:</strong>{' '}
              {result.anxiety_predicted === 'Not evaluated (not depressed)' ? (
                <span className="text-gray-500 italic">{result.anxiety_predicted}</span>
              ) : result.anxiety_predicted === 1 ? (
                <span className="text-red-600 font-semibold">Positive</span>
              ) : (
                <span className="text-green-600 font-semibold">Negative</span>
              )}
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
