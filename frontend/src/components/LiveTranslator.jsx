import React, { useEffect, useRef } from 'react';
import { useTranslationContext } from '../contexts/TranslationContext';
import { API_BASE_URL } from '../config';

// Batch-based client cache
const cacheGet = (from, to, text) => {
  try { return sessionStorage.getItem(`tx_${from}_${to}_${text}`) || null; } catch { return null; }
};
const cacheSet = (from, to, text, translated) => {
  try { sessionStorage.setItem(`tx_${from}_${to}_${text}`, translated); } catch {}
};

async function translateBatch(texts, from, to) {
  // Split into cached and uncached
  const out = new Array(texts.length);
  const toFetch = [];
  const fetchIdx = [];
  texts.forEach((t, i) => {
    if (!t || from === to) {
      out[i] = t;
      return;
    }
    const c = cacheGet(from, to, t);
    if (c !== null) {
      out[i] = c;
    } else {
      toFetch.push(t);
      fetchIdx.push(i);
    }
  });

  if (toFetch.length === 0) return out;
  try {
    const res = await fetch(`${API_BASE_URL}/translate-batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts: toFetch, source: from, target: to }),
    });
    const data = await res.json();
    const arr = Array.isArray(data?.translations) ? data.translations : toFetch;
    arr.forEach((tr, j) => {
      const i = fetchIdx[j];
      out[i] = tr ?? toFetch[j];
      cacheSet(from, to, toFetch[j], out[i]);
    });
  } catch {
    // fallback to original for unfetched
    fetchIdx.forEach((i, j) => { out[i] = toFetch[j]; cacheSet(from, to, toFetch[j], toFetch[j]); });
  }
  return out;
}

// Minimal fallback dictionaries for critical UI labels (extend gradually)
const FALLBACK_DICTIONARIES = {
  hi: new Map([
    ['Smart Meal Recommendations', 'स्मार्ट भोजन सिफारिशें'],
    ['Personalized suggestions based on your profile and preferences', 'आपकी प्रोफ़ाइल और पसंद के आधार पर निजी सुझाव'],
    ['Refresh', 'रीफ़्रेश'],
    ['Food Preference', 'भोजन पसंद'],
    ['All Foods', 'सभी भोजन'],
    ['Vegetarian Only', 'सिर्फ़ शाकाहारी'],
    ['Showing all meal types', 'सभी भोजन प्रकार दिखा रहे हैं'],
    ['All Meals', 'सभी भोजन'],
    ['Breakfast', 'नाश्ता'],
    ['Lunch', 'दोपहर का भोजन'],
    ['Dinner', 'रात का भोजन'],
    ['Snacks', 'नाश्ता/स्नैक्स'],
    ['Meal Suggestions', 'भोजन सुझाव'],
    ['Calories:', 'कैलोरी:'],
    ['Carbs:', 'कार्ब्स:'],
    ['Fiber:', 'फ़ाइबर:'],
    ['Protein:', 'प्रोटीन:'],
    ['Fat:', 'वसा:'],
    ['GL:', 'जीएल:'],
    ['Glycemic Index:', 'ग्लाइसेमिक इंडेक्स:'],
    ['Low GI', 'लो जीआई'],
    ['Risk Score:', 'जोखिम स्कोर:'],
    ['Add to Log', 'लॉग में जोड़ें'],
    ['How recommendations work', 'सिफारिशें कैसे काम करती हैं'],
    ['Our system analyzes your personal health profile, previous meal logs, and response patterns to suggest meals that are most likely to keep your blood sugar levels stable. Recommendations improve as you log more meals.', 'हमारी प्रणाली आपकी स्वास्थ्य प्रोफ़ाइल, पिछले भोजन लॉग और प्रतिक्रिया पैटर्न का विश्लेषण करती है ताकि ऐसे भोजन सुझाए जा सकें जो आपके ब्लड शुगर को स्थिर रखने की अधिक संभावना रखते हैं। जैसे-जैसे आप अधिक भोजन लॉग करेंगे, सिफारिशें बेहतर होती जाएंगी।'],
    ['More Features Coming Soon!', 'और सुविधाएँ जल्द आ रही हैं!'],
    ['Advanced analytics, detailed health reports, integration with wearable devices, and personalized meal planning are on the way.', 'उन्नत विश्लेषण, विस्तृत स्वास्थ्य रिपोर्ट, वियरेबल डिवाइस एकीकरण और निजी भोजन योजना जल्द ही आ रही हैं।'],
  ]),
};

function applyFallback(text, from, to) {
  if (!text || from === to) return text;
  const dict = FALLBACK_DICTIONARIES[to];
  if (!dict) return text;
  // Try exact match
  if (dict.has(text)) return dict.get(text);
  // Handle labels with inline values (e.g., Risk Score: 12%)
  for (const [k, v] of dict.entries()) {
    if (text.startsWith(k)) return text.replace(k, v);
  }
  return text;
}

function walkTextNodes(root, callback) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => {
      const value = node.nodeValue?.trim();
      if (!value) return NodeFilter.FILTER_REJECT;
      // Skip scripts/styles or if parent has data-no-translate
      const parent = node.parentElement;
      if (!parent || parent.closest('[data-no-translate]')) return NodeFilter.FILTER_REJECT;
      if (['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(parent.tagName)) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    }
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach(callback);
}

export default function LiveTranslator({ from = 'en' }) {
  const { language, liveTranslateEnabled } = useTranslationContext();
  const prevLang = useRef(language);
  const observerRef = useRef(null);
  const debounceTimer = useRef(null);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    if (!liveTranslateEnabled) return;
    const to = language;

    let cancelled = false;
    const originals = new WeakMap();

    const translateAll = async () => {
      const nodes = [];
      walkTextNodes(document.body, (node) => {
        const original = node.nodeValue;
        if (!originals.has(node)) originals.set(node, original);
        const baseText = originals.get(node) || original;
        nodes.push({ node, text: baseText });
      });
      // Also translate common attributes like placeholder, title, aria-label
      const attrTargets = [];
      const attrOriginals = new WeakMap();
      const attrEls = Array.from(document.body.querySelectorAll('[placeholder],[title],[aria-label]'));
      attrEls.forEach((el) => {
        ['placeholder', 'title', 'aria-label'].forEach((attr) => {
          const val = el.getAttribute(attr);
          if (val && !el.closest('[data-no-translate]')) {
            let map = attrOriginals.get(el);
            if (!map) { map = {}; attrOriginals.set(el, map); }
            if (!map[attr]) map[attr] = val;
            attrTargets.push({ el, attr, text: map[attr] });
          }
        });
      });
      if (nodes.length === 0 && attrTargets.length === 0) return;
      // Batch by chunks to avoid huge payloads
      const chunkSize = 100;
      const processTexts = async (items, setter) => {
        for (let start = 0; start < items.length && !cancelled; start += chunkSize) {
          const slice = items.slice(start, start + chunkSize);
          const texts = slice.map(n => n.text);
          try {
            let translated = await translateBatch(texts, from, to);
            // Fallback dictionary for key UI phrases if provider returns original
            translated = translated.map((t, i) => t === texts[i] ? applyFallback(texts[i], from, to) : t);
            if (!cancelled) setter(slice, translated);
          } catch {
            const direct = texts.map(t => applyFallback(t, from, to));
            if (!cancelled) setter(slice, direct);
          }
        }
      };

      await processTexts(nodes, (slice, translated) => {
        slice.forEach((n, i) => { n.node.nodeValue = translated[i]; });
      });
      await processTexts(attrTargets, (slice, translated) => {
        slice.forEach((n, i) => { n.el.setAttribute(n.attr, translated[i]); });
      });
    };

    translateAll();
    prevLang.current = to;
    // Observe DOM changes to keep translations live
    if (observerRef.current) {
      try { observerRef.current.disconnect(); } catch {}
      observerRef.current = null;
    }
    if (liveTranslateEnabled) {
      observerRef.current = new MutationObserver(() => {
        if (debounceTimer.current) clearTimeout(debounceTimer.current);
        debounceTimer.current = setTimeout(() => {
          if (!cancelled) translateAll();
        }, 250);
      });
      observerRef.current.observe(document.body, {
        childList: true,
        characterData: true,
        subtree: true,
      });
    }

    return () => {
      cancelled = true;
      if (observerRef.current) {
        try { observerRef.current.disconnect(); } catch {}
        observerRef.current = null;
      }
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
        debounceTimer.current = null;
      }
    };
  }, [language, liveTranslateEnabled, from]);

  return null; // no UI
}
