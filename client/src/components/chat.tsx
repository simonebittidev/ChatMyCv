'use client';

import { useEffect, useRef, useState } from 'react';
import { ArrowUpIcon } from '@heroicons/react/24/solid';

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

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";

    const socket = new WebSocket(`${protocol}://${window.location.host}/ws`);

    console.log(`WebSocket URL: ${protocol}://${window.location.host}/ws`);
    socketRef.current = socket;

    socket.onopen = () => {
      console.log("✅ WebSocket connesso");
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (typeof data === "object" && data !== null && "event" in data) {
          const typingMsg = `${data["agent_name"]} is typing...`;
          setIsTyping(prev => (prev.includes(typingMsg) ? prev : [...prev, typingMsg]));

          // Timeout automatico per rimuovere dopo 3s
          setTimeout(() => {
            setIsTyping(prev => prev.filter(item => item !== typingMsg));
          }, 3000);
        }
        else {
          const messages = JSON.parse(data);
          if (Array.isArray(messages)) {
            const lastElement = messages[messages.length - 1];

            setIsTyping(prev => prev.filter(item => item !== `${lastElement.agent_name} is typing...`));

            console.log(JSON.stringify(messages));
            console.log('messaggi ws ricevuti in formato array');
            setChatMessages(messages);
          } else {
            console.log('messaggi ws non ricevuti in formato array');
            console.log(messages);
          }
        }

      } catch (err) {
        console.error("Errore parsing messaggio:", err);
      }
    };

    socket.onclose = () => {
      console.log("❌ WebSocket disconnesso");
    };

    return () => {
      socket.close();
    };
  });

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [chatMessages]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  const handleSend = () => {
    if (!chatInput.trim()) return;

    let contentToSend = chatInput.trim();

    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ text: contentToSend }));
      setChatInput("");
      setChatMessages([...chatMessages, { role: 'human', content: contentToSend }]);
    }
  };

  const MOBILE_WIDTH = 768;
  const isMobile = typeof window !== "undefined" && window.innerWidth < MOBILE_WIDTH;

  return (
    <div className="flex flex-col h-screen items-center justify-center">
      <div className="flex flex-col h-full w-full max-w-xl mx-auto">
        <div className="h-16 w-full mt-20">
          {!chatMessages?.length &&
            <div className='py-20 px-4'>
              <div className='text-2xl text-gray-800 font-semibold '>Hello there!</div>
              <div className='text-2xl text-zinc-500 '>I'm here to answer any questions you may have about Simone, is there anything in particular you want to know?</div>
            </div>
          }
          <div
            className="flex-1 overflow-y-auto p-4"
            ref={chatBoxRef}
            style={{
              paddingBottom: isInputFocused && isMobile ? "270px" : "150px",
              transition: "padding-bottom 0.2s",
            }}
          >
            {chatMessages.map((msg, i) => (
              <div key={i} className={`mb-5 flex items-start gap-2 ${msg.role === 'human' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`rounded-xl px-5 py-3 text-sm max-w-[80%] whitespace-pre-line ${msg.role === 'human' ? 'bg-black text-white' : 'bg-gray-100 text-black'}`}
                >
                  {msg.content}
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
              <div data-testid="suggested-actions" className="grid sm:grid-cols-2 gap-2 w-full mb-2"><div className="block">
                <button className="inline-flex font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:bg-accent hover:text-accent-foreground text-left border border-gray-300 rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full h-auto justify-start items-start " >
                  <span className="font-medium ">Experiences</span>
                  <span className="text-muted-foreground text-gray-500 break-words">Which companies has Simone worked for?</span>
                </button>
              </div>
                <div className="block">
                  <button className="inline-flex font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50  hover:bg-accent hover:text-accent-foreground text-left border border-gray-300 rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full h-auto justify-start items-start">
                    <span className="font-medium">Technologies</span>
                    <span className="text-muted-foreground text-gray-500 break-words">What technologies does Simone have experience with?</span>
                  </button>
                </div>
                <div className="hidden sm:block">
                  <button className="break-words inline-flex font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:bg-accent hover:text-accent-foreground text-left border border-gray-300 rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full h-auto justify-start items-start">
                    <span className="font-medium">Personal projects</span>
                    <span className="text-muted-foreground text-gray-500">What personal projects has Simone worked on?</span>
                    </button>
                    </div>
                    <div className="hidden sm:block" >
                      <button className="break-words inline-flex font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:bg-accent hover:text-accent-foreground text-left border border-gray-300 rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full h-auto justify-start items-start">
                        <span className="font-medium">Personal life</span>
                      <span className="text-muted-foreground text-gray-500 break-words">Who is Simone?</span>
                  </button>
                </div>
              </div>
            }
            <div tabIndex={0} className="flex gap-2 border p-2 rounded-2xl border-gray-800 bg-gray-100 h-24 focus-within:ring-2 focus-within:ring-gary-700 focus-within:outline-none">
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
                className="bottom-0 right-3 w-8 h-8 bg-gray-500 text-white rounded-full flex items-center justify-center hover:bg-gray-800"
              >
                <ArrowUpIcon className='w-4 h-4'></ArrowUpIcon>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatContent;