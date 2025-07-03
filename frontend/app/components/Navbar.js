'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
  const pathname = usePathname();

  const linkStyle = (path) =>
    `px-4 py-2 rounded-md hover:bg-zinc-700 transition ${
      pathname === path ? 'bg-zinc-700 font-semibold' : 'text-gray-300'
    }`;

  return (
    <nav className="bg-zinc-900 border-b border-zinc-800 w-full py-4 px-6 flex justify-center shadow-sm">
      <div className="flex gap-6">
        <Link href="/" className={linkStyle('/')}>
          🏠 Home
        </Link>
        <Link href="/about" className={linkStyle('/about')}>
          📄 About Project
        </Link>
      </div>
    </nav>
  );
}
