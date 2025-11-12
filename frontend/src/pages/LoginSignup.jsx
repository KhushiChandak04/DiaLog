import React, { useEffect, useState } from 'react';
import { signInWithGoogle, signUpWithEmail, signInWithEmail } from '../services/firebase';
import { useLocation, useNavigate } from 'react-router-dom';
import { T } from '../components/TranslatedText';

const LoginSignup = ({ initialMode }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const location = useLocation();

  // Determine initial mode from route or query string
  useEffect(() => {
    try {
      const path = location.pathname.toLowerCase();
      const params = new URLSearchParams(location.search);
      const hash = (location.hash || '').toLowerCase();
      const explicitMode = (initialMode || params.get('mode') || params.get('tab') || '').toLowerCase();
      const wantsSignup =
        explicitMode === 'signup' ||
        path.endsWith('/signup') ||
        path.endsWith('/register') ||
        hash.includes('signup') ||
        hash.includes('register');
      setIsLogin(!wantsSignup);
    } catch {
      setIsLogin(true);
    }
    // only evaluate on mount and when path/search/hash change
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname, location.search, location.hash, initialMode]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    try {
      if (isLogin) {
        await signInWithEmail(email, password);
      } else {
        await signUpWithEmail(email, password, username);
      }
      navigate('/profile');
    } catch (err) {
      setError(err.message);
    }
  };

  const handleGoogleSignIn = async () => {
    setError('');
    try {
      await signInWithGoogle();
      navigate('/profile');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-primary-50 dark:bg-gray-900">
      <div className="max-w-md w-full bg-white dark:bg-gray-800 rounded-xl shadow-soft p-8">
        <h2 className="text-2xl font-bold mb-6 text-primary-700 dark:text-primary-400 text-center">
          <T>{isLogin ? 'Login' : 'Create an Account'}</T>
        </h2>
        <form onSubmit={handleSubmit} className="space-y-6">
          {!isLogin && (
            <div>
              <label htmlFor="username" className="block text-sm font-medium mb-2 text-neutral-900 dark:text-white"><T>Username</T></label>
              <input
                type="text"
                id="username"
                value={username}
                onChange={e => setUsername(e.target.value)}
                className="w-full px-4 py-3 rounded-xl border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-gray-700 text-neutral-900 dark:text-white placeholder-neutral-400 dark:placeholder-neutral-300"
                required
              />
            </div>
          )}
          <div>
            <label htmlFor="email" className="block text-sm font-medium mb-2 text-neutral-900 dark:text-white"><T>Email</T></label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              className="w-full px-4 py-3 rounded-xl border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-gray-700 text-neutral-900 dark:text-white placeholder-neutral-400 dark:placeholder-neutral-300"
              required
            />
          </div>
          <div>
            <label htmlFor="password" className="block text-sm font-medium mb-2 text-neutral-900 dark:text-white"><T>Password</T></label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                id="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                className="w-full px-4 py-3 pr-12 rounded-xl border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-gray-700 text-neutral-900 dark:text-white placeholder-neutral-400 dark:placeholder-neutral-300"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-200 focus:outline-none"
                title={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? (
                  // Eye-slash icon (password hidden)
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 11-4.243-4.243m4.242 4.242L9.88 9.88" />
                  </svg>
                ) : (
                  // Eye icon (password visible)
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                )}
              </button>
            </div>
          </div>
          {error && <div className="text-danger-600 text-sm text-center">{error}</div>}
          <button
            type="submit"
            className="w-full py-3 rounded-xl bg-primary-600 text-white font-semibold hover:bg-primary-700 transition-all"
          >
            <T>{isLogin ? 'Login' : 'Sign Up'}</T>
          </button>
        </form>
        
        {/* Divider */}
        <div className="mt-6 mb-6">
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-300 dark:border-gray-600"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white dark:bg-gray-800 text-gray-500 dark:text-gray-400">
                <T>Or continue with</T>
              </span>
            </div>
          </div>
        </div>

        {/* Google Sign-In Button */}
        <button
          onClick={handleGoogleSignIn}
          className="w-full flex items-center justify-center px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl shadow-sm bg-white dark:bg-gray-700 text-sm font-medium text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-600 transition-all"
        >
          <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
          </svg>
          <T>Continue with Google</T>
        </button>
        <div className="mt-6 text-center">
          <button
            className="text-primary-600 hover:underline"
            onClick={() => {
              const next = !isLogin;
              setIsLogin(next);
              // reflect mode in URL without page reload for shareable state
              const url = new URL(window.location.href);
              if (next) {
                url.searchParams.delete('mode');
                window.history.replaceState(null, '', url.pathname + (url.search ? url.search : '') + url.hash);
              } else {
                url.searchParams.set('mode', 'signup');
                window.history.replaceState(null, '', url.pathname + '?' + url.searchParams.toString() + url.hash);
              }
            }}
          >
            <T>{isLogin ? "Don't have an account? Sign Up" : "Already have an account? Login"}</T>
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoginSignup;
