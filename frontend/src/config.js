// Global constants placeholder
export const API_BASE_URL =
	process.env.REACT_APP_API_BASE_URL ||
	(typeof window !== 'undefined' ? window.__API_BASE_URL__ : undefined) ||
	(process.env.NODE_ENV === 'production'
		? 'https://dialog-backend.onrender.com'
		: 'http://localhost:8000');
