'use client';

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-zinc-900 text-white px-6 py-10 flex flex-col items-center">
      <div className="max-w-4xl w-full space-y-10">

        {/* 🔹 Hero Section */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">🧠 AI-Powered Mental Health Screening</h1>
          <p className="text-gray-400 text-lg">A research-backed web app that uses natural language to screen for depression and anxiety.</p>
        </div>

        {/* 🔸 Purpose */}
        <Section title="🔍 Purpose">
          Our goal is to detect early signs of <strong>depression</strong> and <strong>anxiety</strong> by analyzing how people naturally express themselves in text. This enables subtle, low-pressure self-screening for mental health concerns.
        </Section>

        {/* 🔸 Problem Statement */}
        <Section title="🎯 Problem Statement">
          Traditional tools often feel clinical or invasive. Many individuals avoid formal screening altogether. This app provides a conversational, self-paced experience to help surface emotional patterns in writing.
        </Section>

        {/* 🔸 Core Functionality */}
        <Section title="⚙️ How It Works">
          <ul className="list-disc list-inside space-y-1 text-gray-300">
            <li>📝 User answers 8 short reflective questions</li>
            <li>📄 Their responses are compiled into a transcript</li>
            <li>🤖 Backend model extracts 399 features (lexical, emotional, embeddings)</li>
            <li>📊 AI predicts depression & anxiety presence</li>
          </ul>
        </Section>

        {/* 🔸 Research Progress */}
        <Section title="📊 Research Status">
          <ul className="list-disc list-inside space-y-1 text-gray-300">
            <li>✅ 119 labeled transcripts used for training</li>
            <li>✅ Features: Sentence-BERT, Empath, NLP tokens</li>
            <li>✅ Depression model: Random Forest</li>
            <li>✅ Anxiety model: Voting Classifier</li>
            <li>📉 Current depression accuracy: ~58%</li>
          </ul>
        </Section>

        {/* 🔸 What's Next */}
        <Section title="🚀 Future Roadmap">
          <ul className="list-disc list-inside space-y-1 text-gray-300">
            <li>📈 Add PHQ-8 score prediction + severity grading</li>
            <li>🌍 Train on clinical, multilingual data</li>
            <li>📊 Show emotion analysis + progress chart</li>
            <li>🔁 Enable anonymous mood tracking over time</li>
          </ul>
        </Section>

        {/* 🔸 Tech Stack */}
        <Section title="🧰 Tech Stack">
          <ul className="grid grid-cols-2 gap-y-2 gap-x-4 text-gray-300">
            <li><strong>Frontend:</strong> Next.js, Tailwind CSS</li>
            <li><strong>Backend:</strong> FastAPI (Python)</li>
            <li><strong>ML:</strong> scikit-learn, Empath, S-BERT</li>
            <li><strong>Deployment:</strong> Vercel + Render</li>
          </ul>
        </Section>
      </div>
    </main>
  );
}

// 🔧 Reusable section block
function Section({ title, children }) {
  return (
    <section className="bg-zinc-800 p-6 rounded-2xl shadow border border-zinc-700">
      <h2 className="text-2xl font-semibold mb-3 text-white">{title}</h2>
      <div className="text-gray-300 leading-relaxed">{children}</div>
    </section>
  );
}
