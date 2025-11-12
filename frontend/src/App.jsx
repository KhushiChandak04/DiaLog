import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Profile from './pages/Profile';
// ...existing code...
import LoginSignup from './pages/LoginSignup';
import MealLog from './pages/MealLog';
import Dashboard from './pages/Dashboard';
import Feedback from './pages/Feedback';
import NotFound from './pages/NotFound';
import './tailwind.css';
import FoodSafety from './pages/FoodSafety';
import { TranslationProvider } from './contexts/TranslationContext';
import LiveTranslator from './components/LiveTranslator';
import { auth } from './services/firebase';
import AiAssistantButton from './components/AiAssistantButton';
import AiChatPanel from './components/AiChatPanel';

function ProtectedRoute({ children }) {
  const [authState, setAuthState] = React.useState({ loading: true, user: null });

  React.useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((u) => {
      setAuthState({ loading: false, user: u });
    });
    return () => unsubscribe();
  }, []);

  if (authState.loading) {
    return (
      <div className="w-full py-16 flex items-center justify-center text-primary-600 dark:text-primary-400">
        Loading...
      </div>
    );
  }

  if (!authState.user) {
    return <Navigate to="/" replace />;
  }
  return children;
}

function App() {
  const [isAssistantOpen, setIsAssistantOpen] = React.useState(false);
  const [authUser, setAuthUser] = React.useState(null);

  React.useEffect(() => {
    const unsub = auth.onAuthStateChanged((u) => setAuthUser(u));
    return () => unsub();
  }, []);
  return (
    <Router>
      <TranslationProvider>
      <LiveTranslator />
      <div className="flex flex-col min-h-screen">
        {/* Navigation */}
        <Navbar />
        {/* Global AI Assistant (only for authenticated users) */}
        {authUser && (
          <>
            <AiAssistantButton onClick={() => setIsAssistantOpen(true)} />
            <AiChatPanel isOpen={isAssistantOpen} onClose={() => setIsAssistantOpen(false)} />
          </>
        )}
        
        {/* Main Content */}
        <main className="flex-grow">
          <Routes>
            {/* Home Page */}
            <Route path="/" element={<Home />} />
            
            {/* Profile Page */}
            <Route path="/profile" element={<Profile />} />
            
            {/* Login/Signup Page */}
            <Route path="/auth" element={<LoginSignup />} />
            <Route path="/login" element={<LoginSignup initialMode="login" />} />
            <Route path="/signup" element={<LoginSignup initialMode="signup" />} />
            
            {/* Meal Logging */}
            <Route path="/meal-log" element={<MealLog />} />
            
            {/* Dashboard (protected) */}
            <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />

            {/* Safety & Nutrition (protected) */}
            <Route path="/safety" element={<ProtectedRoute><FoodSafety /></ProtectedRoute>} />
            
            {/* Feedback */}
            <Route path="/feedback" element={<Feedback />} />
            
            {/* Unknown routes → go to landing page */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        
        {/* Footer */}
        <Footer />
      </div>
      </TranslationProvider>
    </Router>
  );
}


export default App;
