'use client';

export default function Question({ question, index, value, onChange }) {
  return (
    <div className="mb-6">
      <label className="block mb-2 font-medium text-gray-200">
        {index + 1}. {question}
      </label>
      <textarea
        className="w-full p-4 rounded-lg bg-zinc-800 text-white border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-white placeholder-gray-400 transition"
        rows={3}
        value={value}
        onChange={(e) => onChange(index, e.target.value)}
        placeholder="Type your response here..."
      />
    </div>
  );
}
