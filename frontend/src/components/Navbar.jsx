import React, { useState, useEffect } from 'react';
import { auth } from '../services/firebase';
import { useNavigate } from 'react-router-dom';
import { Link, useLocation } from 'react-router-dom';
import { 
  Bars3Icon, 
  XMarkIcon, 
  SunIcon, 
  MoonIcon
} from '@heroicons/react/24/outline';
import LogoPlaceholder from './LogoPlaceholder';
import LanguageSwitcher from './LanguageSwitcher';
import { T } from './TranslatedText';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Check localStorage and system preference
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('darkMode');
      if (saved !== null) {
        return JSON.parse(saved);
      }
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  });
  const location = useLocation();

  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(setUser);
    return () => unsubscribe();
  }, []);

  const navigation = user
    ? [
        { name: 'Dashboard', href: '/dashboard' },
        { name: 'Log Meal', href: '/meal-log' },
        { name: 'Safety & Nutrition', href: '/safety' },
        { name: 'Profile', href: '/profile' }
      ]
    : [
        { name: 'Home', href: '/' }
      ];

  // Apply dark mode to document
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', JSON.stringify(isDarkMode));
  }, [isDarkMode]);

  const isActive = (path) => location.pathname === path;

  const toggleTheme = () => {
    setIsDarkMode(prev => !prev);
  };

  return (
    <nav className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-b border-neutral-200 dark:border-neutral-700 sticky top-0 z-50 transition-all duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo and brand */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-3 group">
              <div className="relative">
                <LogoPlaceholder className="w-8 h-8 transition-transform duration-200 group-hover:scale-110" isDark={isDarkMode} />
                <div className="absolute inset-0 bg-primary-500/20 rounded-full scale-0 group-hover:scale-125 transition-transform duration-300"></div>
              </div>
              <span className="text-xl font-bold text-primary-600 dark:text-primary-400">
                DiaLog
              </span>
            </Link>
          </div>

          {/* Desktop navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navigation.map((item) => (
              <Link
                key={item.name}
                to={item.href}
                className={`
                  relative px-4 py-2 rounded-xl text-sm font-medium transition-all duration-300 group
                  ${isActive(item.href)
                    ? 'text-primary-700 bg-primary-50 dark:text-primary-300 dark:bg-primary-900/30 shadow-sm'
                    : 'text-neutral-600 dark:text-neutral-300 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-primary-50/50 dark:hover:bg-primary-900/20'
                  }
                `}
              >
                <T>{item.name}</T>
                {isActive(item.href) && (
                  <div className="absolute inset-x-0 -bottom-1 h-0.5 bg-primary-500 rounded-full"></div>
                )}
              </Link>
            ))}
            {/* Language switcher */}
            <LanguageSwitcher />

            {/* Auth button (Login/Logout) */}
            {user ? (
              <button
                onClick={async () => {
                  await auth.signOut();
                  navigate('/');
                }}
                className="relative px-4 py-2 rounded-xl text-sm font-medium bg-danger-50 text-danger-700 dark:bg-danger-900/30 dark:text-danger-300 hover:bg-danger-100 dark:hover:bg-danger-900/40 transition-all duration-300 mr-2"
              >
                <T>Log Out</T>
              </button>
            ) : (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => navigate('/login')}
                  className="relative px-4 py-2 rounded-xl text-sm font-medium bg-success-50 text-success-700 dark:bg-success-900/30 dark:text-success-300 hover:bg-success-100 dark:hover:bg-success-900/40 transition-all duration-300"
                >
                  <T>Log In</T>
                </button>
                <button
                  onClick={() => navigate('/signup')}
                  className="relative px-4 py-2 rounded-xl text-sm font-medium bg-primary-600 text-white hover:bg-primary-700 transition-all duration-300"
                >
                  <T>Sign Up</T>
                </button>
              </div>
            )}
            {/* Enhanced Theme toggle */}
            <button
              onClick={toggleTheme}
              className="relative p-3 rounded-xl bg-primary-50 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 hover:bg-primary-100 dark:hover:bg-primary-900/40 hover:shadow-md transition-all duration-300 group"
              aria-label="Toggle theme"
            >
              <div className="relative z-10">
                {isDarkMode ? (
                  <SunIcon className="h-5 w-5 transform group-hover:rotate-12 transition-transform duration-300" />
                ) : (
                  <MoonIcon className="h-5 w-5 transform group-hover:-rotate-12 transition-transform duration-300" />
                )}
              </div>
            </button>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center space-x-2">
            {/* Mobile language switcher */}
            <LanguageSwitcher />
            {/* Mobile theme toggle */}
            <button
              onClick={toggleTheme}
              className="relative p-2.5 rounded-lg bg-primary-50 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 hover:bg-primary-100 dark:hover:bg-primary-900/40 transition-all duration-300"
              aria-label="Toggle theme"
            >
              {isDarkMode ? (
                <SunIcon className="h-5 w-5" />
              ) : (
                <MoonIcon className="h-5 w-5" />
              )}
            </button>
            
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="inline-flex items-center justify-center p-2.5 rounded-lg text-neutral-600 dark:text-neutral-300 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-primary-50/50 dark:hover:bg-primary-900/20 transition-all duration-300"
              aria-label="Toggle menu"
            >
              {isOpen ? (
                <XMarkIcon className="h-6 w-6" />
              ) : (
                <Bars3Icon className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Enhanced Mobile menu */}
      <div className={`md:hidden overflow-hidden transition-all duration-300 ${isOpen ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'}`}>
        <div className="bg-white/95 dark:bg-gray-900/95 backdrop-blur-lg border-t border-neutral-200 dark:border-neutral-700">
          <div className="px-4 pt-4 pb-6 space-y-2">
            {navigation.map((item) => (
              <Link
                key={item.name}
                to={item.href}
                className={`
                  block px-4 py-3 rounded-xl text-base font-medium transition-all duration-300
                  ${isActive(item.href)
                    ? 'text-primary-700 bg-primary-100 dark:text-primary-300 dark:bg-primary-900/40 shadow-sm'
                    : 'text-neutral-700 dark:text-neutral-300 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-primary-50 dark:hover:bg-primary-900/20'
                  }
                `}
                onClick={() => setIsOpen(false)}
              >
                <T>{item.name}</T>
              </Link>
            ))}
            {user ? (
              <button
                onClick={async () => {
                  await auth.signOut();
                  navigate('/');
                  setIsOpen(false);
                }}
                className="block w-full px-4 py-3 rounded-xl text-base font-medium bg-danger-50 text-danger-700 dark:bg-danger-900/30 dark:text-danger-300 hover:bg-danger-100 dark:hover:bg-danger-900/40 transition-all duration-300"
              >
                <T>Log Out</T>
              </button>
            ) : (
              <button
                onClick={() => {
                  navigate('/auth');
                  setIsOpen(false);
                }}
                className="block w-full px-4 py-3 rounded-xl text-base font-medium bg-success-50 text-success-700 dark:bg-success-900/30 dark:text-success-300 hover:bg-success-100 dark:hover:bg-success-900/40 transition-all duration-300"
              >
                <T>Log In</T>
              </button>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
