'use client'

export default function Navbar() {

  return (
    <header className="fixed top-0 left-0 w-full z-50 bg-white">
      <nav aria-label="Global" className="flex items-center justify-between p-6 lg:px-8">
        <div className="flex lg:flex-1">
          <a href="/" className="-m-1.5 p-1.5">
            <h1 className="text-2xl font-bold text-gray-700 italic">Ask my cv</h1>
          </a>
        </div>
      </nav>
    </header>
  )
}
