import type { Metadata } from "next";
import "./globals.css";
import Footer from "@/components/footer";
import CookieWidget from "@/components/cookie-widget";

export const metadata: Metadata = {
  title: "Ask my cv - Simone Bitti",
  description: "Ask my cv is a web application that allows you to interact with my CV using natural language. You can ask questions about my skills, experience, and education, and get answers in real-time.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  
  return (
    <html lang="en" className="h-full">
      <head>
          <link rel="icon" href="favicon.ico" type="icon"/>
          <meta charSet="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          {/* <CookieWidget/> */}
      </head>
      <body className="min-h-screen flex flex-col">
        {children}
      </body>
    </html>
  );
}
