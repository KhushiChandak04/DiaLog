// Centralized health calculation utilities (BMI & categories)
// Ensures consistent logic across Profile, recommendations, and safety components.

/**
 * Convert height string/value + unit to meters
 * Supports: cm, ft (decimal feet)
 */
export function heightToMeters(value, unit) {
  const h = parseFloat(value);
  if (!h || h <= 0) return null;
  if (unit === 'cm') return h / 100;
  if (unit === 'ft') return h * 0.3048; // decimal feet -> meters
  return null;
}

/**
 * Convert weight string/value + unit to kilograms
 * Supports: kg, lbs
 */
export function weightToKg(value, unit) {
  const w = parseFloat(value);
  if (!w || w <= 0) return null;
  if (unit === 'kg') return w;
  if (unit === 'lbs') return w * 0.453592;
  return null;
}

/**
 * Calculate BMI with robust guards
 * Returns null if insufficient data or obviously invalid ranges
 */
export function calculateBMI(weightValue, weightUnit, heightValue, heightUnit) {
  const m = heightToMeters(heightValue, heightUnit);
  const kg = weightToKg(weightValue, weightUnit);
  if (m === null || kg === null) return null;
  if (m < 0.8 || m > 2.8) return null; // sanity guard
  if (kg < 15 || kg > 400) return null;
  const bmi = kg / (m * m);
  return Number(bmi.toFixed(1));
}

/**
 * Classify BMI per WHO ranges
 */
export function classifyBMI(bmi) {
  if (bmi === null || bmi === undefined || isNaN(bmi)) return { category: 'Unknown', color: 'text-gray-500 bg-gray-100 dark:text-gray-400 dark:bg-gray-800' };
  if (bmi < 18.5) return { category: 'Underweight', color: 'text-blue-600 bg-blue-100 dark:text-blue-400 dark:bg-blue-900/30' };
  if (bmi < 25) return { category: 'Normal', color: 'text-success-600 bg-success-100 dark:text-success-400 dark:bg-success-900/30' };
  if (bmi < 30) return { category: 'Overweight', color: 'text-warning-600 bg-warning-100 dark:text-warning-400 dark:bg-warning-900/30' };
  return { category: 'Obese', color: 'text-danger-600 bg-danger-100 dark:text-danger-400 dark:bg-danger-900/30' };
}

/**
 * Convenience bundle (optional)
 */
export function computeBMIBundle({ height, heightUnit, weight, weightUnit }) {
  const bmi = calculateBMI(weight, weightUnit, height, heightUnit);
  return { bmi, ...classifyBMI(bmi) };
}
