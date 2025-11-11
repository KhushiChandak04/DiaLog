import { API_BASE_URL } from '../config';

/**
 * Fetch all available foods from the database
 * @returns {Promise<{foods: string[], count: number}>}
 */
export async function fetchFoods() {
  try {
    const response = await fetch(`${API_BASE_URL}/foods`);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to fetch foods: ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error fetching foods:", error);
    throw error;
  }
}

/**
 * Get details about a specific food
 * @param {string} foodName - The name of the food
 * @returns {Promise<Object>} - Food details
 */
export async function getFoodDetails(foodName) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/food/${encodeURIComponent(foodName)}`
    );

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Failed to get food details");
    }

    const data = await response.json();
    // Normalize: flatten common fields to top-level for older components
    const flattened = {
      name: data.name,
      ...data,
      calories: data.calories ?? data?.nutritional_info?.calories_kcal,
      carbs: data.carbs ?? data?.nutritional_info?.carbs_g,
      protein: data.protein ?? data?.nutritional_info?.protein_g,
      fat: data.fat ?? data?.nutritional_info?.fat_g,
      fiber: data.fiber ?? data?.nutritional_info?.fiber_g,
      glycemicIndex: data.glycemicIndex ?? data?.nutritional_info?.glycemic_index,
      glycemic_load: data.glycemic_load ?? data?.nutritional_info?.glycemic_load
    };
    return flattened;
  } catch (error) {
    console.error(`Error fetching details for ${foodName}:`, error);
    throw error;
  }
}

/**
 * Predict if a meal is diabetes-friendly
 * @param {Object} mealData - User and meal data
 * @returns {Promise<Object>} - Prediction results
 */
export async function predictDiabetesFriendly(mealData) {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(mealData),
    });

    if (!response.ok) {
      let detail = 'Unknown error';
      try {
        const errorData = await response.json();
        detail = typeof errorData?.detail === 'string' ? errorData.detail : JSON.stringify(errorData?.detail ?? errorData);
      } catch (e) {
        const text = await response.text();
        detail = text || 'Unknown error';
      }
      throw new Error(`Prediction error: ${detail}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Prediction request failed:", error);
    throw error;
  }
}

/**
 * Predict safety for multiple meals at once
 * @param {Object} mealData - User data with array of meals
 * @returns {Promise<Object>} - Prediction results for multiple meals
 */
export async function predictMultipleMeals(mealData) {
  try {
    const response = await fetch(`${API_BASE_URL}/predict-multiple`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(mealData),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Multiple meal prediction error: ${errorData.detail || "Unknown error"}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Multiple meal prediction request failed:", error);
    // Fallback: if multiple meal endpoint doesn't exist yet, process meals individually
    if (error.message.includes('404') || error.message.includes('Not Found')) {
      console.warn("Multiple meal endpoint not available, processing meals individually...");
      return await processMealsIndividually(mealData);
    }
    throw error;
  }
}

/**
 * Fallback function to process multiple meals individually
 * @param {Object} mealData - User data with array of meals
 * @returns {Promise<Object>} - Combined prediction results
 */
async function processMealsIndividually(mealData) {
  try {
    const predictions = [];
    let overallRisk = 'low';
    let isAnySafe = true;

    for (const meal of mealData.meals) {
      const individualMealData = {
        age: mealData.age,
        gender: mealData.gender,
        weight_kg: mealData.weight_kg,
        height_cm: mealData.height_cm,
        fasting_sugar: mealData.fasting_sugar,
        post_meal_sugar: mealData.post_meal_sugar,
        meal_taken: meal.meal_taken,
        time_of_day: mealData.time_of_day,
        portion_size: meal.portion_size,
        portion_unit: meal.portion_unit
      };

      const prediction = await predictDiabetesFriendly(individualMealData);
      predictions.push({
        meal: meal.meal_taken,
        ...prediction
      });

      // Determine overall risk level
      if (prediction.risk_level === 'high' || prediction.risk_level === 'unsafe') {
        overallRisk = 'high';
        isAnySafe = false;
      } else if (prediction.risk_level === 'moderate' || prediction.risk_level === 'medium') {
        if (overallRisk !== 'high') overallRisk = 'moderate';
      }

      if (!prediction.is_safe) isAnySafe = false;
    }

    // Combine results
    const combinedRecommendations = [];
    predictions.forEach(pred => {
      if (pred.recommendations) {
        combinedRecommendations.push(...pred.recommendations);
      }
    });

    return {
      is_safe: isAnySafe,
      risk_level: overallRisk,
      message: `Analysis complete for ${predictions.length} meal(s). ${isAnySafe ? 'Overall combination looks safe.' : 'Some meals may need attention.'}`,
      confidence: predictions.reduce((acc, pred) => acc + (pred.confidence || 0), 0) / predictions.length,
      predictions: predictions,
      recommendations: combinedRecommendations.slice(0, 5), // Limit recommendations
      bmi: predictions[0]?.bmi || 0
    };
  } catch (error) {
    console.error("Error processing meals individually:", error);
    throw error;
  }
}

/**
 * Convert backend food details into display rows of [label, value]
 */
export function buildNutritionRows(foodDetails) {
  if (!foodDetails) return [];
  const n = foodDetails.nutritional_info || {};
  const rows = [
    ['Calories (kcal)', n.calories_kcal],
    ['Carbs (g)', n.carbs_g],
    ['Protein (g)', n.protein_g],
    ['Fat (g)', n.fat_g],
    ['Fiber (g)', n.fiber_g],
    ['Sugar (g)', n.sugar_g],
    ['Glycemic Index', n.glycemic_index],
    ['Glycemic Load', n.glycemic_load],
    ['Sodium (mg)', n.sodium_mg],
    ['Serving Size (g)', n.serving_size_g],
    ['Default Portion', n.default_portion],
  ];
  return rows.filter(([_, v]) => v !== undefined && v !== null && v !== '');
}

/**
 * Get personalized meal recommendations using ML model
 * @param {Object} userProfile - User profile data
 * @returns {Promise<Object>} - ML-powered recommendations
 */
export async function getPersonalizedRecommendations(userProfile) {
  try {
    const response = await fetch(`${API_BASE_URL}/recommendations`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(userProfile),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Recommendation error: ${errorData.detail || "Unknown error"}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Recommendation request failed:", error);
    throw error;
  }
}

/**
 * Get truly personalized meal recommendations using individual user ML models
 * @param {Object} userProfile - User profile data with user_id
 * @returns {Promise<Object>} - Personalized ML recommendations based on user logs
 */
export async function getTrulyPersonalizedRecommendations(userProfile) {
  try {
    const response = await fetch(`${API_BASE_URL}/truly-personalized-recommendations`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(userProfile),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Personalized recommendation error: ${errorData.detail || "Unknown error"}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Personalized recommendation request failed:", error);
    // Fallback to general recommendations
    return await getPersonalizedRecommendations(userProfile);
  }
}

/**
 * Check if the API is healthy and models are loaded
 * @returns {Promise<Object>} - Health status
 */
export async function checkApiHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
      throw new Error("API health check failed");
    }

    return await response.json();
  } catch (error) {
    console.error("Health check failed:", error);
    throw error;
  }
}
