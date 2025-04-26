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
import Providers from "@/components/Providers";
import Navbar from "@/components/Navbar";

export const metadata = {
  title: "Plagiarism Detection & AI Content Detection",
  description: "QuickDetect",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="antialiased">
        <Providers>
          <Navbar />
          <div className="mt-32">
            { children }
          </div>
        </Providers>
        <Toaster />
      </body>
    </html>
  );
}
