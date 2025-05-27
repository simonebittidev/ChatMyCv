'use client'

import Navbar from '@/components/navbar';
import Chat from '@/components/chat';

export default function Home() {
  return (
    <>
      <div className="bg-white">
        <Chat></Chat>
      </div>
    </>
  )
}