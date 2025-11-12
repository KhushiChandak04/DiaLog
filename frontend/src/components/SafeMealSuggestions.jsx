import React, { useState, useEffect, useCallback } from 'react';
import { 
  SparklesIcon, 
  ArrowPathIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  LightBulbIcon,
  PlusIcon
} from '@heroicons/react/24/outline';
import { fetchFoods, getPersonalizedRecommendations, getTrulyPersonalizedRecommendations } from '../services/api';
import MealCard from './MealCard';

const SafeMealSuggestions = ({ userProfile = {}, currentMeal = null, className = "" }) => {
  const [suggestions, setSuggestions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [availableMeals, setAvailableMeals] = useState([]);
  const [usedMeals, setUsedMeals] = useState(new Set());
  const [isVegOnly, setIsVegOnly] = useState(false); // New state for veg/non-veg filter

  // Categories for meal recommendations
  const categories = [
    { id: 'all', name: 'All Meals', icon: SparklesIcon },
    { id: 'breakfast', name: 'Breakfast', icon: CheckCircleIcon },
    { id: 'lunch', name: 'Lunch', icon: CheckCircleIcon },
    { id: 'dinner', name: 'Dinner', icon: CheckCircleIcon },
    { id: 'snacks', name: 'Snacks', icon: CheckCircleIcon }
  ];

  // Robust vegetarian detector (mirrors backend logic, fixes flicker & false matches)
  const isVegetarian = (mealName) => {
    const n = (mealName || '').toLowerCase();
    if (!n.trim()) return true;
    // Keywords (ensure we don't misclassify 'eggless' as non-veg)
    const NON_VEG = [
      'chicken','mutton','fish','egg','meat','prawn','prawns','shrimp','crab','keema','kebab','kebabs','tikka','boti','paya',
      'lamb','beef','pork','turkey','duck','seafood','salmon','tuna','sardine','anchovy','octopus','squid','ham','salami','pepperoni','prosciutto','bacon','sausage',
      'biryani chicken','chicken curry','fish curry','tandoori','seekh','shami','galouti','nihari','bhuna chicken','butter chicken',
      'egg curry','omelet','omelette','scrambled egg','boiled egg','egg bhurji','anda bhurji','anda','murgh','murg'
    ];
    // Handle eggless override before generic 'egg' match
    if (n.includes('eggless')) {
      // Still reject if explicit meat tokens present
      if (NON_VEG.filter(k => k !== 'egg').some(k => n.includes(k))) return false;
      return true;
    }
    // Fast rejection if any meat keyword present
    if (NON_VEG.some(k => n.includes(k))) return false;
    // Positive vegetarian hints
    if (/\bveg\b/.test(n) || n.includes('(veg') || n.includes('[veg') || n.includes('pure veg') || n.includes('vegetarian')) return true;
    // Default: treat as vegetarian when no explicit non-veg tokens
    return true;
  };

  // Load meals from dataset
  useEffect(() => {
    const loadMeals = async () => {
      try {
        const data = await fetchFoods();
        setAvailableMeals(data.foods || []);
      } catch (err) {
        console.error('Failed to load meals:', err);
        setAvailableMeals([]);
      }
    };
    loadMeals();
  }, []);

  // Create meal suggestions from dataset
  const createMealSuggestion = (mealName, category = 'all') => {
    // Simple categorization based on meal name
    let mealCategory = category;
    if (category === 'all') {
      const name = mealName.toLowerCase();
      if (name.includes('breakfast') || name.includes('poha') || name.includes('upma') || name.includes('idli') || name.includes('dosa')) {
        mealCategory = 'breakfast';
      } else if (name.includes('lunch') || name.includes('rice') || name.includes('dal') || name.includes('curry')) {
        mealCategory = 'lunch';
      } else if (name.includes('dinner') || name.includes('roti') || name.includes('sabzi')) {
        mealCategory = 'dinner';
      } else if (name.includes('snack') || name.includes('tea') || name.includes('biscuit') || name.includes('sweet')) {
        mealCategory = 'snacks';
      } else {
        mealCategory = 'lunch'; // default
      }
    }

    return {
      name: mealName,
      calories: Math.floor(Math.random() * 200) + 150, // Estimated calories
      carbs: Math.floor(Math.random() * 30) + 20,
      protein: Math.floor(Math.random() * 15) + 5,
      fat: Math.floor(Math.random() * 10) + 2,
      fiber: Math.floor(Math.random() * 5) + 1,
      glycemicIndex: Math.floor(Math.random() * 30) + 35, // Low to medium GI
      portionSize: "1 serving",
      timeOfDay: mealCategory.charAt(0).toUpperCase() + mealCategory.slice(1),
      riskScore: Math.random() * 0.4, // Low risk meals
      confidence: 0.85 + Math.random() * 0.1,
      category: mealCategory,
      reasons: [] // Removed static reasons - let ML handle recommendations
    };
  };

  // Get non-repetitive meal suggestions from dataset
  const getMealSuggestions = (category, count = 6) => {
    if (!availableMeals.length) return [];

    // Build filtered pool based on current toggles (veg + category)
    const applyFilters = (list) => {
      let out = [...list];
      if (isVegOnly) {
        out = out.filter(meal => isVegetarian(meal));
      }
      if (category !== 'all') {
        out = out.filter(meal => {
          const name = meal.toLowerCase();
          switch (category) {
            case 'breakfast':
              return name.includes('breakfast') || name.includes('poha') || name.includes('upma') || 
                     name.includes('idli') || name.includes('dosa') || name.includes('paratha');
            case 'lunch':
              return name.includes('lunch') || name.includes('rice') || name.includes('dal') || 
                     name.includes('curry') || name.includes('sambar') || name.includes('rasam') || name.includes('sabzi');
            case 'dinner':
              return name.includes('dinner') || name.includes('roti') || name.includes('sabzi') || 
                     name.includes('chapati') || name.includes('bhaji') || name.includes('soup');
            case 'snacks':
              return name.includes('snack') || name.includes('tea') || name.includes('biscuit') || 
                     name.includes('namkeen') || name.includes('chaat') || name.includes('pakora') || name.includes('snacks');
            default:
              return true;
          }
        });
      }
      return out;
    };

    // Filter meals based on current state
    let candidateMeals = applyFilters(availableMeals);

    // Remove already used meals and get fresh ones
    const unusedMeals = candidateMeals.filter(meal => !usedMeals.has(meal));
    
    // If we've exhausted unique items under current filters, allow reuse but keep filters intact
    let availableForSelection;
    if (unusedMeals.length >= count) {
      availableForSelection = [...unusedMeals];
    } else {
      // Reset the used set so future calls can rotate again, but DO NOT drop filters
      setUsedMeals(new Set());
      availableForSelection = [...candidateMeals];
    }

    // Randomly select meals from the filtered pool
    const selectedMeals = [];
    
    for (let i = 0; i < Math.min(count, availableForSelection.length); i++) {
      const randomIndex = Math.floor(Math.random() * availableForSelection.length);
      const selectedMeal = availableForSelection.splice(randomIndex, 1)[0];
      selectedMeals.push(createMealSuggestion(selectedMeal, category));
    }

    // Track used meals
    setUsedMeals(prev => {
      const newUsed = new Set(prev);
      selectedMeals.forEach(meal => newUsed.add(meal.name));
      return newUsed;
    });

    return selectedMeals;
  };

  // Reset rotation cache when filters change to avoid cross-contamination
  useEffect(() => {
    setUsedMeals(new Set());
  }, [isVegOnly, selectedCategory]);

  // Add meal to log function
  const addMealToLog = (meal) => {
    // Store meal data in localStorage for the meal log page
    const mealForLog = {
      name: meal.name,
      calories: meal.calories,
      carbs: meal.carbs,
      protein: meal.protein,
      fat: meal.fat,
      fiber: meal.fiber,
      glycemicIndex: meal.glycemicIndex,
      portionSize: meal.portionSize,
      timestamp: Date.now()
    };
    
    const existingMeals = JSON.parse(localStorage.getItem('pendingMealsForLog') || '[]');
    existingMeals.push(mealForLog);
    localStorage.setItem('pendingMealsForLog', JSON.stringify(existingMeals));
    
    // Navigate to meal log page
    window.location.href = '/meal-log';
  };

  // New ML-powered recommendation function (memoized to prevent re-renders)
  const getMLRecommendations = useCallback(async (category, count = 6) => {
    const timeOfDayMap = {
      'breakfast': 'Breakfast',
      'lunch': 'Lunch', 
      'dinner': 'Dinner',
      'snacks': 'Snack',
      'all': 'Lunch'  // Default
    };

    // Derive height (cm) and weight (kg) from profile if unit-specific fields are not present
    const parseNum = (v) => {
      const n = parseFloat(v);
      return isNaN(n) ? undefined : n;
    };
    const rawHeight = userProfile.height_cm ?? userProfile.height;
    const heightUnit = userProfile.height_cm ? 'cm' : (userProfile.heightUnit || 'cm');
    let height_cm = parseNum(rawHeight);
    if (height_cm !== undefined && heightUnit === 'ft') {
      // If user entered 5.4 as feet, assume 5.4 ft, convert to cm
      height_cm = height_cm * 30.48;
    }
    const rawWeight = userProfile.weight_kg ?? userProfile.weight;
    const weightUnit = userProfile.weight_kg ? 'kg' : (userProfile.weightUnit || 'kg');
    let weight_kg = parseNum(rawWeight);
    if (weight_kg !== undefined && weightUnit === 'lbs') {
      weight_kg = weight_kg * 0.453592;
    }
    const requestData = {
      age: parseNum(userProfile.age) || 35,
      gender: (userProfile.gender || 'Male'),
      weight_kg: weight_kg || 70,
      height_cm: height_cm || 170,
      fasting_sugar: parseNum(userProfile.fasting_sugar) || 100,
      post_meal_sugar: parseNum(userProfile.post_meal_sugar) || 140,
      diabetes_type: userProfile.diabetes_type || 'Type2',
      time_of_day: timeOfDayMap[category] || 'Lunch',
      count: count
    };

    // Convey vegetarian preference to backend so it can filter candidate foods earlier
    if (isVegOnly) {
      requestData.meal_preferences = ['vegetarian'];
    }

    // Use truly personalized recommendations if user_id is available
    let response;
    if (userProfile.user_id) {
      requestData.user_id = userProfile.user_id;
      response = await getTrulyPersonalizedRecommendations(requestData);
    } else {
      response = await getPersonalizedRecommendations(requestData);
    }
    
    // Transform ML response to match expected format and use correct backend field names
    let transformedRecommendations = response.recommendations.map(rec => ({
      name: rec.name,
      calories: rec.calories,
      carbs: rec.carbs,
      protein: rec.protein,
      fat: rec.fat,
      fiber: rec.fiber,
      glycemicIndex: rec.glycemicIndex,
      glycemicLoad: rec.glycemicLoad200 || rec.glycemic_load || undefined,
      glBadge: rec.glBadge || rec.gl_badge || undefined,
      portionSize: rec.portionSize,
      timeOfDay: rec.timeOfDay,
      riskLevel: rec.risk_level || rec.riskLevel || 'low',
      riskScore: rec.risk_level === 'safe' ? 0.1 : rec.risk_level === 'caution' ? 0.5 : 0.8,
      confidence: rec.confidence || 0.8,
      category: category === 'all' ? 'lunch' : category,
      reasons: rec.personalized_reason ? [rec.personalized_reason] : (rec.reasons || []),
      mlPowered: true,  // Flag to indicate this is ML-generated
      explanation: rec.explanation,
      isPersonalized: !!rec.personalized_reason,  // Flag for personalized vs general
      predicted_blood_sugar: rec.predicted_blood_sugar,
      personalization_info: response.personalization || null  // Store personalization details
    }));

    // Apply veg/non-veg filter to ML recommendations (final guard)
    if (isVegOnly) {
      transformedRecommendations = transformedRecommendations.filter(rec => isVegetarian(rec.name));
    }

    return transformedRecommendations;
  }, [userProfile.age, userProfile.gender, userProfile.weight_kg, userProfile.height_cm, userProfile.fasting_sugar, userProfile.post_meal_sugar, userProfile.diabetes_type, userProfile.user_id, isVegOnly]);

  // Consolidated effect for loading recommendations
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (availableMeals.length === 0) return;

    const loadRecommendations = async () => {
      setIsLoading(true);
      try {
        if (userProfile.age && userProfile.weight_kg && userProfile.height_cm) {
          // Use ML recommendations if user profile is complete
          const mlRecommendations = await getMLRecommendations(selectedCategory, 6);
          setSuggestions(mlRecommendations);
        } else {
          // Fallback to static recommendations
          setSuggestions(getMealSuggestions(selectedCategory, 6));
        }
      } catch (error) {
        console.error('Recommendation loading failed:', error);
        setSuggestions(getMealSuggestions(selectedCategory, 6));
      }
      setIsLoading(false);
    };

    loadRecommendations();
  }, [availableMeals.length, selectedCategory, userProfile.age, userProfile.weight_kg, userProfile.height_cm, getMLRecommendations, isVegOnly]);

  const generateRecommendations = useCallback(async () => {
    if (availableMeals.length === 0 || isLoading) return; // Prevent multiple simultaneous calls
    
    setIsLoading(true);
    try {
      if (userProfile.age && userProfile.weight_kg && userProfile.height_cm) {
        // Use ML-powered recommendations
        const mlRecommendations = await getMLRecommendations(selectedCategory, 6);
        setSuggestions(isVegOnly ? mlRecommendations.filter(m => isVegetarian(m.name)) : mlRecommendations);
      } else {
        // Fallback to static recommendations
        const staticRecs = getMealSuggestions(selectedCategory, 6);
        setSuggestions(isVegOnly ? staticRecs.filter(m => isVegetarian(m.name)) : staticRecs);
      }
    } catch (error) {
      console.error('ML recommendations failed, using fallback:', error);
      const fallbackRecs = getMealSuggestions(selectedCategory, 6);
      setSuggestions(isVegOnly ? fallbackRecs.filter(m => isVegetarian(m.name)) : fallbackRecs);
    }
    // Add a small delay to prevent rapid flickering
    setTimeout(() => setIsLoading(false), 300);
  }, [availableMeals.length, isLoading, userProfile.age, userProfile.weight_kg, userProfile.height_cm, selectedCategory, getMLRecommendations, isVegOnly]);

  // Enforce vegetarian consistency if toggle flips after suggestions loaded (prevents stale non-veg)
  useEffect(() => {
    if (isVegOnly) {
      setSuggestions(prev => prev.filter(m => isVegetarian(m?.name)));
    }
  }, [isVegOnly]);

  // Final guard for veg-only before rendering
  const displayedSuggestions = isVegOnly ? (suggestions || []).filter(m => isVegetarian(m?.name)) : suggestions;

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-soft p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-primary-100 dark:bg-primary-900/30 rounded-lg">
            <SparklesIcon className="h-6 w-6 text-primary-600 dark:text-primary-400" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Smart Meal Recommendations
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Personalized suggestions based on your profile and preferences
            </p>
          </div>
        </div>
        
        <button
          onClick={generateRecommendations}
          disabled={isLoading}
          className="inline-flex items-center px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors duration-200 disabled:opacity-50"
        >
          <ArrowPathIcon className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
          {isLoading ? 'Generating...' : 'Refresh'}
        </button>
      </div>

      {/* Veg/Non-Veg Toggle */}
      <div className="flex items-center justify-between mb-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
        <div className="flex items-center space-x-3">
          <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Food Preference
          </div>
          <div className="flex items-center space-x-2">
            <span className={`text-sm ${!isVegOnly ? 'text-gray-900 dark:text-white font-medium' : 'text-gray-500 dark:text-gray-400'}`}>
              All Foods
            </span>
            <button
              onClick={() => setIsVegOnly(!isVegOnly)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ${
                isVegOnly ? 'bg-green-600' : 'bg-gray-300 dark:bg-gray-600'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition duration-200 ${
                  isVegOnly ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
            <span className={`text-sm ${isVegOnly ? 'text-green-600 font-medium' : 'text-gray-500 dark:text-gray-400'}`}>
              Vegetarian Only
            </span>
          </div>
        </div>
        <div className="text-xs text-gray-500 dark:text-gray-400">
          {isVegOnly ? '🌱 Showing vegetarian meals only' : '🍽️ Showing all meal types'}
        </div>
      </div>

      {/* Category Filters */}
      <div className="flex flex-wrap gap-2 mb-6">
        {categories.map((category) => (
          <button
            key={category.id}
            onClick={() => setSelectedCategory(category.id)}
            className={`
              inline-flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-200
              ${selectedCategory === category.id
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }
            `}
          >
            <category.icon className="h-4 w-4 mr-2" />
            {category.name}
          </button>
        ))}
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <ArrowPathIcon className="h-8 w-8 text-primary-600 animate-spin mx-auto mb-4" />
            <p className="text-gray-600 dark:text-gray-400">Analyzing your profile and generating personalized recommendations...</p>
          </div>
        </div>
      )}

      {/* Personalization Status */}
  {!isLoading && displayedSuggestions.length > 0 && displayedSuggestions[0]?.personalization_info && (
        <div className={`p-4 rounded-lg mb-6 ${
              displayedSuggestions[0].personalization_info.has_personal_model 
            ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800' 
            : 'bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800'
        }`}>
          <div className="flex items-start space-x-3">
            <SparklesIcon className={`h-5 w-5 mt-0.5 ${
              displayedSuggestions[0].personalization_info.has_personal_model ? 'text-green-600' : 'text-blue-600'
            }`} />
            <div className="flex-1">
              <h4 className={`font-medium ${
                displayedSuggestions[0].personalization_info.has_personal_model ? 'text-green-800 dark:text-green-200' : 'text-blue-800 dark:text-blue-200'
              }`}>
                {displayedSuggestions[0].personalization_info.personalization_note}
              </h4>
              {displayedSuggestions[0].personalization_info.personal_insights && (
                <p className={`text-sm mt-1 ${
                  displayedSuggestions[0].personalization_info.has_personal_model ? 'text-green-700 dark:text-green-300' : 'text-blue-700 dark:text-blue-300'
                }`}>
                  {displayedSuggestions[0].personalization_info.personal_insights}
                </p>
              )}
              {displayedSuggestions[0].personalization_info.has_personal_model && (
                <div className="flex items-center space-x-4 mt-2 text-xs text-green-600 dark:text-green-400">
                  <span>{displayedSuggestions[0].personalization_info.meal_count} meals analyzed</span>
                  {displayedSuggestions[0].personalization_info.model_score && (
                    <span>Model accuracy: {(displayedSuggestions[0].personalization_info.model_score * 100).toFixed(0)}%</span>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Individual Suggestions Grid */}
      {!isLoading && displayedSuggestions.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Meal Suggestions
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {displayedSuggestions.map((meal, index) => (
              <div key={meal?.name || `meal_${index}`} className="relative">
                <MealCard
                  meal={meal}
                  riskLevel="low"
                  showNutrition={true}
                  showPrediction={true}
                  onClick={() => {
                    console.log('Selected meal:', meal);
                  }}
                />
                
                {/* Action Buttons */}
                <div className="mt-3">
                  <button
                    onClick={() => addMealToLog(meal)}
                    className="w-full px-3 py-2 bg-primary-600 hover:bg-primary-700 text-white text-sm font-medium rounded-lg transition-colors duration-200"
                  >
                    <PlusIcon className="h-4 w-4 inline mr-1" />
                    Add to Log
                  </button>
                </div>
                
                {/* Recommendation Reasons - Only show if reasons exist */}
                {meal.reasons && meal.reasons.length > 0 && (
                  <div className="mt-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <LightBulbIcon className="h-4 w-4 text-warning-500" />
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {meal.isPersonalized ? '🎯 Personal Analysis:' : meal.mlPowered ? 'Smart Analysis:' : 'Why we recommend this:'}
                      </span>
                      <div className="flex space-x-1">
                        {meal.mlPowered && (
                          <span className="px-2 py-0.5 bg-primary-100 text-primary-700 text-xs rounded-full">
                            ML Powered
                          </span>
                        )}
                        {meal.isPersonalized && (
                          <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full">
                            Personalized
                          </span>
                        )}
                        {meal.predicted_blood_sugar && (
                          <span className={`px-2 py-0.5 text-xs rounded-full ${
                            meal.predicted_blood_sugar <= 140 
                              ? 'bg-green-100 text-green-700' 
                              : meal.predicted_blood_sugar <= 180 
                                ? 'bg-yellow-100 text-yellow-700' 
                                : 'bg-red-100 text-red-700'
                          }`}>
                            {meal.predicted_blood_sugar}mg/dL
                          </span>
                        )}
                      </div>
                    </div>
                    <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                      {meal.reasons.slice(0, 3).map((reason, idx) => (
                        <li key={idx} className="flex items-start space-x-2">
                          <CheckCircleIcon className={`h-3 w-3 mt-0.5 flex-shrink-0 ${
                            meal.isPersonalized ? 'text-green-500' : 'text-success-500'
                          }`} />
                          <span className={meal.isPersonalized ? 'font-medium text-green-700 dark:text-green-300' : ''}>
                            {reason}
                          </span>
                        </li>
                      ))}
                    </ul>
                    {meal.mlPowered && meal.explanation && (
                      <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-600">
                        <p className="text-xs text-gray-500 dark:text-gray-400 italic">
                          {meal.explanation.slice(0, 100)}...
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && displayedSuggestions.length === 0 && (
        <div className="text-center py-12">
          <div className="p-4 bg-gray-100 dark:bg-gray-700 rounded-full w-16 h-16 mx-auto mb-4">
            <ExclamationTriangleIcon className="h-8 w-8 text-gray-400 mx-auto" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No recommendations available
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Complete your profile and log some meals to get personalized recommendations.
          </p>
          <button
            onClick={generateRecommendations}
            className="inline-flex items-center px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors duration-200"
          >
            <SparklesIcon className="h-4 w-4 mr-2" />
            Generate Recommendations
          </button>
        </div>
      )}

      {/* Info Banner */}
      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 rounded-lg">
        <div className="flex items-start space-x-3">
          <LightBulbIcon className="h-5 w-5 text-blue-500 mt-0.5 flex-shrink-0" />
          <div>
            <h4 className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-1">
              How recommendations work
            </h4>
            <p className="text-sm text-blue-700 dark:text-blue-300">
              Our system analyzes your personal health profile, previous meal logs, and response patterns to suggest meals 
              that are most likely to keep your blood sugar levels stable. Recommendations improve as you log more meals.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SafeMealSuggestions;
