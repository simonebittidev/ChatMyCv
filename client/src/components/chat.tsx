'use client';

import { useEffect, useRef, useState } from 'react';
import { ArrowUpIcon, SparklesIcon } from '@heroicons/react/24/solid';
import ReactMarkdown from 'react-markdown';

const suggestedActions = [
  {
    label: "Experiences",
    text: "Which companies has Simone worked for?",
  },
  {
    label: "Technologies",
    text: "What technologies does Simone have experience with?",
  },
  {
    label: "Personal projects",
    text: "What personal projects has Simone worked on?",
  },
  {
    label: "Contact",
    text: "How can I contact Simone?",
  }
];

function getRandomItems<T>(arr: T[], n: number): T[] {
  const shuffled = arr.slice().sort(() => 0.5 - Math.random());
  return shuffled.slice(0, n);
}

type ChatMessage = {
  role: 'human' | 'ai';
  content: string;
};

const ChatContent = () => {
  const chatBoxRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isInputFocused, setInputFocused] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const [aiError, setAiError] = useState(false);

  const handleSuggestedClick = (text: string) => {
      console.log("handling click");
      setChatMessages(prev => [
        ...prev,
        { role: 'human', content: text },
        { role: 'ai', content: '' }
      ]);
      const history = [...chatMessages, { role: 'human', content: text }];
      startStream(text, history);
  };

  const startStream = (text: string, historyMessages: ChatMessage[]) => {
    setAiError(false);

    const encodedHistory = encodeURIComponent(JSON.stringify(historyMessages.slice(-3)));
    const eventSource = new EventSource(`/stream?text=${encodeURIComponent(text)}&history=${encodedHistory}`);
  
    let currentMessage = "";
  
    eventSource.onmessage = (event) => {

      console.log(event.data);
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

  const handleSend = () => {

    if (!chatInput.trim()) return;
  
    const text = chatInput.trim();
    setChatMessages((prev) => [...prev, { role: 'human', content: text }, { role: 'ai', content: '' }]);
    setChatInput('');
    const history = [...chatMessages, { role: 'human', content: text }];
    startStream(text, history);
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  const MOBILE_WIDTH = 768;
  const isMobile = typeof window !== "undefined" && window.innerWidth < MOBILE_WIDTH;

  // Initialize suggested actions only once on first render
  const [actionsToShow] = useState<{
    label: string;
    text: string;
  }[]>(() => {
    return isMobile
      ? getRandomItems(suggestedActions, 2)
      : suggestedActions;
  });

  return (
    <div className="flex flex-col h-screen items-center justify-center">
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
              paddingBottom: isInputFocused && isMobile ? "220px" : "124px",
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

                {msg.role === 'ai' && msg.content === '[ERROR]' ? (
                  <div className='rounded-xl text-sm whitespace-pre-line text-red-700 mt-1'>
                      An error occurred with the service. Please try again later.
                  </div>
                ): (
                <div className={`rounded-xl text-sm whitespace-pre-line ${msg.role === 'human' ? 'bg-black text-white px-5 max-w-[80%] py-3' : 'text-gray-700'}`}>
                  <ReactMarkdown>
                    {msg.content}
                  </ReactMarkdown>
                </div>
                )}
              </div>
            ))}

            <div ref={messagesEndRef} />
          </div>
        </div>
        <div className="p-3 bg-white fixed bottom-0 left-0 right-0 z-20 w-full flex justify-center">
          <div className="w-full max-w-xl">
            {!chatMessages?.length && (
              <div data-testid="suggested-actions" className="grid sm:grid-cols-2 gap-2 w-full mb-2 items-stretch">
                {actionsToShow.map((action, i) => (
                  <div className={`block h-full ${!isMobile && i > 1 ? 'hidden sm:block' : ''}`} key={action.label}>
                    <button
                      onClick={() => handleSuggestedClick(action.text)}
                      className="flex flex-col font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:bg-accent hover:text-accent-foreground text-left border border-gray-300 rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full h-full justify-start items-start"
                    >
                      <span className="font-medium">{action.label}</span>
                      <span className="text-muted-foreground text-gray-500 break-words">{action.text}</span>
                    </button>
                  </div>
                ))}
              </div>
            )}
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
                This is a personal study project; answers may be inaccurate.<br/>
              </span>
            </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatContent;