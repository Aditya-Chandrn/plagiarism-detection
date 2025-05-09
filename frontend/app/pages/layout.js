// import "./globals.css";
// import { Toaster } from "@/components/ui/toaster";

// export const metadata = {
//   title:
//     "A plagiarism detection website which detects AI generated content and Text Simiilarity",
//   description: "QuickDetect",
// };

// export default function RootLayout({ children }) {
//   return (
//     <html lang="en">
//       <body className={`antialiased`}>
//         {children}
//         <Toaster />
//       </body>
//     </html>
//   );
// }

import "../globals.css";
import { Toaster } from "@/components/ui/toaster";
import Navbar from "@/components/Navbar";

export const metadata = {
  title: "Plagiarism Detection & AI Content Detection",
  description: "QuickDetect",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="antialiased">
          <Navbar />
          <div className="mt-32">
            { children }
          </div>
        <Toaster />
      </body>
    </html>
  );
}
