import "./globals.css";

export const metadata = {
  title:
    "A plagiarism detection website which detects AI generated content and Text Simiilarity",
  description: "QuickDetect",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`antialiased`}>{children}</body>
    </html>
  );
}
