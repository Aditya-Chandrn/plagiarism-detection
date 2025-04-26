"use client"
import Link from "next/link";
import { Button } from "@/components/ui/button";
import axios from "axios";
import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";

const Navbar = () => {
  const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;
  const [displayName, setDisplayName] = useState("");
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    async function fetchUser() {
      try {
        const res = await axios.get(`${BACKEND_URL}/user/me`, {
          withCredentials: true,
        });
        if (res.data && res.data.display_name) {
          setDisplayName(res.data.display_name);
        }
      } catch (error) {
        console.error("Error fetching user info:", error);
      }
    }
    fetchUser();
  }, [BACKEND_URL, pathname]);

  const handleLogout = async () => {
    const confirmed = window.confirm("Are you sure you want to log out?");
    if (confirmed) {
      try {
        // Call the backend logout endpoint to remove the HttpOnly cookie.
        await axios.post(`${BACKEND_URL}/user/logout`, null, {
          withCredentials: true,
        });
        // Clear the display name.
        setDisplayName("");
        // Redirect to login page.
        router.push("/auth/login");
      } catch (error) {
        console.error("Error logging out:", error);
      }
    }
  };


  return (
    <header className="fixed top-0 left-0 right-0 bg-white z-50">
      <div className="container mx-auto px-6">
        <div className="h-20 flex items-center justify-between">
          <Link href="/" className="flex items-center space-x-3">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className="h-8 w-8"
            >
              <path d="M12 4L4 8l8 4 8-4-8-4z" />
              <path d="M4 12l8 4 8-4" />
              <path d="M4 16l8 4 8-4" />
            </svg>
            <span className="text-xl font-semibold">PlagiarismShield</span>
          </Link>

          <nav className="flex items-center space-x-12">
            {/* <Link href="/" className="text-base hover:text-gray-600">
              Features
            </Link>
            <Link href="/" className="text-base hover:text-gray-600">
              Pricing
            </Link> */}
            {/* <Link href="/" className="text-base hover:text-gray-600">
              About
            </Link> */}
            <Link href="/auth/login" className="text-base hover:text-gray-600">
              <Button variant="outline" className="text-base font-normal" asChild>
                <Link href="/pages/submission">Paper Upload</Link>
              </Button>
              <Button variant="outline" className="text-base font-normal" asChild>
                <Link href="/pages/submission/summary">Report Summary</Link>
              </Button>
              { displayName ? (
                // If logged in, display the user's name.
                <span className="text-base font-bold text-indigo-600 underline hover:text-indigo-800 cursor-pointer"
                  onClick={ handleLogout }
                  title="Click to logout"
                >
                  { displayName }
                </span>
              ) : (
                <Button variant="outline" className="text-base font-normal" asChild>
                  <Link href="/auth/login">Sign In</Link>
                </Button>
              ) }
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
