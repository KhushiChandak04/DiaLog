// Renders React into HTML
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// If we were redirected from /404.html back to /index.html, restore original route
(() => {
	try {
		const params = new URLSearchParams(window.location.search);
		const redirect = params.get('redirect');
		if (redirect) {
			const allowed = ['/', '/profile', '/auth', '/meal-log', '/dashboard', '/safety', '/feedback'];
			const url = new URL(redirect, window.location.origin);
			const candidate = url.pathname + url.search + url.hash;
			const isAllowed = allowed.some(prefix => candidate === prefix || candidate.startsWith(prefix + (prefix.endsWith('/') ? '' : '/')));
			const finalPath = isAllowed ? candidate : '/';
			window.history.replaceState(null, '', finalPath);
		}
	} catch (e) {
		// Fallback hard redirect to landing page on any parsing errors
		try { window.history.replaceState(null, '', '/'); } catch (_) {}
	}
})();

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
