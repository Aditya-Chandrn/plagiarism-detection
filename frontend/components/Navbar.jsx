import Link from "next/link";
import { Button } from "@/components/ui/button";

const Navbar = () => {
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
            <Link href="/" className="text-base hover:text-gray-600">
              Features
            </Link>
            <Link href="/" className="text-base hover:text-gray-600">
              Pricing
            </Link>
            <Link href="/" className="text-base hover:text-gray-600">
              About
            </Link>
            <Link href="/login" className="text-base hover:text-gray-600">
              <Button variant="outline" className="text-base font-normal">
                Sign In
              </Button>
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
