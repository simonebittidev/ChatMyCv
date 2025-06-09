'use client';

import { useEffect, useRef, useState } from 'react';
import { ArrowUpIcon, SparklesIcon } from '@heroicons/react/24/solid';
import ReactMarkdown from 'react-markdown';

type ChatMessage = {
  role: 'human' | 'ai';
  content: string;
};

const ChatContent = () => {
  const chatBoxRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const socketRef = useRef<WebSocket | null>(null);
  const [isTyping, setIsTyping] = useState<string[]>([]);
  const [isInputFocused, setInputFocused] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const typeAiMessage = (message: string) => {
    const words = message.split(" ");
    let i = 0;
  
    setChatMessages(prev => [
      ...prev,
      { role: 'ai', content: '' }
    ]);
  
    function typeNextWord() {
      setChatMessages(prev => {
        const lastIdx = prev.length - 1;
        const lastMsg = prev[lastIdx];
        if (lastMsg.role !== 'ai') return prev;
  
        const updatedMsg = {
          ...lastMsg,
          // Unisce tutte le parole fino all'indice i, aggiungendo uno spazio tra loro
          content: words.slice(0, i + 1).join(" ")
        };
        const updated = [...prev.slice(0, lastIdx), updatedMsg];
  
        i++;
  
        // Condizione di fine typing
        if (i < words.length) {
          // Intervallo casuale tra una parola e l'altra
          const randomInterval = Math.floor(Math.random() * 80) + 20;
          setTimeout(typeNextWord, randomInterval);
        }
        return updated;
      });
    }
  
    typeNextWord();
  };

  const handleSuggestedClick = (text: string) => {
      console.log("handling click");
      setChatMessages(prev => [
        ...prev,
        { role: 'human', content: text },
        { role: 'ai', content: '' }
      ]);
      startStream(text);
  };

  const startStream = (text: string) => {
    const encodedHistory = encodeURIComponent(JSON.stringify(chatMessages.slice(-3)));
    const eventSource = new EventSource(`/stream?text=${encodeURIComponent(text)}&history=${encodedHistory}`);
  
    let currentMessage = "";
  
    eventSource.onmessage = (event) => {
      if (event.data === "[DONE]") {
        eventSource.close();
        return;
      }
  
      const data = JSON.parse(event.data);
      currentMessage += data.content;
  
      setChatMessages((prev) => {
        const updated = [...prev];
        const lastMsg = updated[updated.length - 1];
        if (lastMsg?.role === 'ai') {
          updated[updated.length - 1] = {
            ...lastMsg,
            content: currentMessage,
          };
          return updated;
        }
        return [...updated, { role: "ai", content: data.content }];
      });
    };
  
    eventSource.onerror = () => {
      console.error("Errore nella ricezione SSE");
      eventSource.close();
    };
  };

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatMessages, isTyping]);

  const handleSend = () => {
    if (!chatInput.trim()) return;
  
    const text = chatInput.trim();
    setChatMessages((prev) => [...prev, { role: 'human', content: text }, { role: 'ai', content: '' }]);
    setChatInput('');
    startStream(text);
  };

  const MOBILE_WIDTH = 768;
  const isMobile = typeof window !== "undefined" && window.innerWidth < MOBILE_WIDTH;

  return (
    <div className="flex flex-col h-screen items-center justify-center mt-10">
      <div className="flex flex-col h-full w-full max-w-xl mx-auto">
        <div className="h-16 w-full mt-5 p-3">
          {!chatMessages?.length &&
            <div className='py-20 px-4'>
              <div className='text-2xl text-gray-800 font-semibold '>Hello there!</div>
              <div className='text-1xl text-zinc-500 '> I’m here to help you explore Simone’s professional profile.<br></br>
                Interested in his experience, skills, or what sets him apart?<br></br>
                Just ask a question—I’ll guide you through his CV to find the answers you need.</div>
            </div>
          }
          <div
            className="flex-1 overflow-y-auto"
            ref={chatBoxRef}
            style={{
              paddingBottom: isInputFocused && isMobile ? "200px" : "104px",
              // transition: "padding-bottom 0.2s",
            }}
          >
            {chatMessages.map((msg, i) => (
              <div key={i} className={`mb-5 flex items-start gap-2 ${msg.role === 'human' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'ai' && (
                  <div className="flex-shrink-0 w-8 h-8 mr-2 flex items-center justify-center border border-gray-200 rounded-full text-gray-800 self-start">
                    <SparklesIcon
                      className={`w-4 h-4 transition-all duration-700 ${
                        !msg.content ? 'animate-pulse' : ''
                      }`}
                    />
                  </div>
                )}
                <div
                  className={`rounded-xl text-sm whitespace-pre-line ${msg.role === 'human' ? 'bg-black text-white px-5 max-w-[80%] py-3' : 'text-gray-700'}`}
                >
                  <ReactMarkdown>
                    {msg.content}
                  </ReactMarkdown>
                </div>
              </div>
            ))}

            {isTyping.map((typing, i) => (
              <div key={`typing-${i}`} className="text-gray-500 text-sm px-10">
                {typing}
              </div>
            ))}

            <div ref={messagesEndRef} />
          </div>
        </div>
        <div className="p-3 bg-white fixed bottom-0 left-0 right-0 z-20 w-full flex justify-center">
          <div className="w-full max-w-xl">
            {!chatMessages?.length &&
              <div data-testid="suggested-actions" className="grid sm:grid-cols-2 gap-2 w-full mb-2">
                <div className="block">
                <button
                  onClick={() => handleSuggestedClick("Which companies has Simone worked for?")}
                  className="flex flex-col  font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:bg-accent hover:text-accent-foreground text-left border border-gray-300 rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full h-auto justify-start items-start " >
                  <span className="font-medium ">Experiences</span>
                  <span className="text-muted-foreground text-gray-500 break-words">Which companies has Simone worked for?</span>
                </button>
              </div>
                <div className="block">
                  <button 
                    onClick={() => handleSuggestedClick("What technologies does Simone have experience with?")}
                    className="flex flex-col font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50  hover:bg-accent hover:text-accent-foreground text-left border border-gray-300 rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full h-auto justify-start items-start">
                    <span className="font-medium">Technologies</span>
                    <span className="text-muted-foreground text-gray-500 break-words">What technologies does Simone have experience with?</span>
                  </button>
                </div>
                <div className="hidden sm:block">
                  <button 
                    onClick={() => handleSuggestedClick("What personal projects has Simone worked on?")}
                    className="break-words flex flex-col font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:bg-accent hover:text-accent-foreground text-left border border-gray-300 rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full h-auto justify-start items-start">
                    <span className="font-medium">Personal projects</span>
                    <span className="text-muted-foreground text-gray-500">What personal projects has Simone worked on?</span>
                    </button>
                    </div>
                    <div className="hidden sm:block" >
                      <button 
                        onClick={() => handleSuggestedClick("How can I contact Simone?")}
                        className="h-full break-words flex flex-col font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:bg-accent hover:text-accent-foreground text-left border border-gray-300 rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full justify-start items-start">
                        <span className="font-medium">Contact</span>
                      <span className="text-muted-foreground text-gray-500 break-words">How can I contact Simone?</span>
                  </button>
                </div>
              </div>
            }
            <div tabIndex={0}>
            <div className="flex gap-2 p-4 rounded-2xl bg-gray-100 h-24 border border-gray-200">
              <textarea
                ref={inputRef}
                value={chatInput}
                onChange={(e) => {
                  const value = e.target.value;
                  setChatInput(value);
                }}
                placeholder="Write your message..."
                onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); handleSend(); } }}
                className="flex-1 hover:bg-gray-100 focus:outline-none"
                onFocus={() => {
                  setInputFocused(true);
                  inputRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
                }}
                onBlur={() => {
                  setInputFocused(false);
                  setTimeout(() => {
                    inputRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
                  }, 100);
                }}
              />
              <button
                onClick={handleSend}
                className="bottom-0 right-3 w-8 h-8 bg-gray-800 text-white rounded-full flex items-center justify-center hover:bg-gray-900"
              >
                <ArrowUpIcon className='w-4 h-4'></ArrowUpIcon>
              </button>
            </div>
            <div className="mt-2 text-xs text-gray-500 text-center w-full">
              <span>
                This is a personal project for study purposes. Answers may contain errors or inaccurate information.<br/>
                For feedback or info: <a href="mailto:simone.bitti.dev@gmail.com" className="underline">simone.bitti.dev@gmail.com</a>
              </span>
            </div>
            </div>
          </div>
        </div>
      </div>
      {/* Informational message below the textbox */}
      
    </div>
  );
};

export default ChatContent;