import React, { useMemo, useState } from 'react';
import { useTranslationContext } from '../contexts/TranslationContext';
import { auth } from '../services/firebase';
import { saveUserProfile, fetchUserProfile } from '../services/firebase';
import { 
  UserIcon, 
  ScaleIcon,
  ArrowPathIcon,
  CheckIcon
} from '@heroicons/react/24/outline';
import LanguageSwitcher from '../components/LanguageSwitcher';
import { T } from '../components/TranslatedText';
import { calculateBMI, classifyBMI } from '../utils/health';

const Profile = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    age: '',
    gender: '',
    diabetes_type: 'Type2',
    height: '',
    weight: '',
    heightUnit: 'cm',
    weightUnit: 'kg'
  });
  const [bmi, setBmi] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [validationErrors, setValidationErrors] = useState({
    age: '',
    height: '',
    weight: ''
  });

  const validateField = (name, value, unitsOverride = {}) => {
    let error = '';
    const currentHeightUnit = unitsOverride.heightUnit ?? formData.heightUnit;
    const currentWeightUnit = unitsOverride.weightUnit ?? formData.weightUnit;
    
    switch (name) {
      case 'age':
        if (value === '') {
          error = 'Age is required';
        } else if (isNaN(value) || value < 10) {
          // Clamp later; keep message friendly for diabetic context
          error = 'Age must be 10+ years';
        } else if (value > 120) {
          error = 'Age must be ≤ 120';
        }
        break;
        
      case 'height':
        if (value === '') {
          error = 'Height is required';
        } else if (isNaN(value) || parseFloat(value) <= 0) {
          error = 'Height must be positive';
        } else {
          const v = parseFloat(value);
          const unit = currentHeightUnit;
          if (unit === 'cm') {
            if (v < 100) error = 'Height seems too low (min 100 cm)';
            else if (v > 250) error = 'Height seems too high (max 250 cm)';
          } else if (unit === 'ft') {
            if (v < 3) error = 'Height seems too low (min 3 ft)';
            else if (v > 8.2) error = 'Height seems too high (max 8.2 ft)';
          }
        }
        break;
        
      case 'weight':
        if (value === '') {
          error = 'Weight is required';
        } else if (isNaN(value) || parseFloat(value) <= 0) {
          error = 'Weight must be positive';
        } else {
          const v = parseFloat(value);
          const unit = currentWeightUnit;
          if (unit === 'kg') {
            if (v < 20) error = 'Weight seems too low (min 20 kg)';
            else if (v > 300) error = 'Weight seems too high (max 300 kg)';
          } else if (unit === 'lbs') {
            if (v < 44) error = 'Weight seems too low (min 44 lbs)';
            else if (v > 660) error = 'Weight seems too high (max 660 lbs)';
          }
        }
        break;
    }
    
    return error;
  };

  const { language } = useTranslationContext();
  // Fetch profile data on mount
  React.useEffect(() => {
    const fetchProfile = async () => {
      const user = auth.currentUser;
      if (user) {
        let profile = await fetchUserProfile(user.uid);
        if (profile) {
          const newFormData = {
            ...profile,
            diabetes_type: profile.diabetes_type || 'Type2',
            email: user.email || profile.email || '',
            name: profile.name || user.displayName || profile.username || ''
          };
          setFormData(newFormData);
          
          // Validate existing data
          const newErrors = {};
          ['age', 'height', 'weight'].forEach(field => {
            const error = validateField(field, newFormData[field] || '', {
              heightUnit: newFormData.heightUnit,
              weightUnit: newFormData.weightUnit,
            });
            if (error) newErrors[field] = error;
          });
          setValidationErrors(newErrors);
          // Preferred language is handled globally via navbar
        } else {
          setFormData(prev => ({
            ...prev,
            email: user.email || ''
          }));
        }
      }
    };
    fetchProfile();
  }, []);

  // Calculate BMI whenever height or weight changes, using centralized util to avoid drift
  React.useEffect(() => {
    // Only recompute when the actual numeric inputs or units change, not on unrelated form fields
    const { height, weight, heightUnit, weightUnit } = formData;
    const computed = calculateBMI(weight, weightUnit, height, heightUnit);
    setBmi(computed);
  }, [formData.height, formData.weight, formData.heightUnit, formData.weightUnit]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;

    // Handle unit conversions when unit selectors change
    if (name === 'heightUnit') {
      setFormData(prev => {
        let height = prev.height;
        // Convert existing numeric value to new unit
        const n = parseFloat(height);
        let converted = height;
        if (!isNaN(n)) {
          if (value === 'cm' && prev.heightUnit === 'ft') {
            converted = (n * 30.48).toFixed(1); // ft -> cm
          } else if (value === 'ft' && prev.heightUnit === 'cm') {
            converted = (n / 30.48).toFixed(2); // cm -> ft
          }
        }
        const next = { ...prev, heightUnit: value, height: converted };
        // Re-validate after conversion
        const err = validateField('height', next.height, { heightUnit: value });
        setValidationErrors(pe => ({ ...pe, height: err }));
        return next;
      });
      return;
    }

    if (name === 'weightUnit') {
      setFormData(prev => {
        let weight = prev.weight;
        const n = parseFloat(weight);
        let converted = weight;
        if (!isNaN(n)) {
          if (value === 'kg' && prev.weightUnit === 'lbs') {
            converted = (n * 0.453592).toFixed(1); // lbs -> kg
          } else if (value === 'lbs' && prev.weightUnit === 'kg') {
            converted = (n / 0.453592).toFixed(1); // kg -> lbs
          }
        }
        const next = { ...prev, weightUnit: value, weight: converted };
        const err = validateField('weight', next.weight, { weightUnit: value });
        setValidationErrors(pe => ({ ...pe, weight: err }));
        return next;
      });
      return;
    }

    // Normal field updates
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));

    // Validate the field and update errors
    if (['age', 'height', 'weight'].includes(name)) {
      const error = validateField(name, value);
      setValidationErrors(prev => ({
        ...prev,
        [name]: error
      }));
    }
  };

  // Clamp/sanitize values on blur to enforce constraints
  const handleBlur = (e) => {
    const { name, value } = e.target;
    let v = value;
    if (name === 'age') {
      let n = parseInt(v || '');
      if (isNaN(n)) n = '';
      else {
        if (n < 10) n = 10;
        if (n > 120) n = 120;
      }
      setFormData(prev => ({ ...prev, age: n }));
      const err = validateField('age', n);
      setValidationErrors(prev => ({ ...prev, age: err }));
    }
    if (name === 'height') {
      let n = parseFloat(v || '');
      if (isNaN(n)) n = '';
      else {
        if (formData.heightUnit === 'cm') {
          if (n < 100) n = 100;
          if (n > 250) n = 250;
        } else {
          if (n < 3) n = 3;
          if (n > 8.2) n = 8.2;
        }
      }
      // Round to sensible precision for the unit
      const rounded = n === '' ? '' : Number(n).toFixed(formData.heightUnit === 'cm' ? 1 : 2);
      setFormData(prev => ({ ...prev, height: rounded }));
      const err = validateField('height', n, { heightUnit: formData.heightUnit });
      setValidationErrors(prev => ({ ...prev, height: err }));
    }
    if (name === 'weight') {
      let n = parseFloat(v || '');
      if (isNaN(n)) n = '';
      else {
        if (formData.weightUnit === 'kg') {
          if (n < 20) n = 20;
          if (n > 300) n = 300;
        } else {
          if (n < 44) n = 44;
          if (n > 660) n = 660;
        }
      }
      const roundedW = n === '' ? '' : Number(n).toFixed(1);
      setFormData(prev => ({ ...prev, weight: roundedW }));
      const err = validateField('weight', n, { weightUnit: formData.weightUnit });
      setValidationErrors(prev => ({ ...prev, weight: err }));
    }
  };

  const bmiCategory = useMemo(() => classifyBMI(bmi), [bmi]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Final validation check before submission
    const finalErrors = {};
    ['age', 'height', 'weight'].forEach(field => {
      const error = validateField(field, formData[field] || '', {
        heightUnit: formData.heightUnit,
        weightUnit: formData.weightUnit,
      });
      if (error) finalErrors[field] = error;
    });
    
    if (Object.keys(finalErrors).length > 0) {
      setValidationErrors(finalErrors);
      return;
    }
    
    // Clamp all values to safe ranges prior to save
    const normalized = { ...formData };
    try {
      let ageN = parseInt(normalized.age);
      if (!isNaN(ageN)) {
        if (ageN < 10) ageN = 10;
        if (ageN > 120) ageN = 120;
        normalized.age = ageN;
      }
      let hN = parseFloat(normalized.height);
      if (!isNaN(hN)) {
        if (normalized.heightUnit === 'cm') { hN = Math.min(Math.max(hN, 100), 250); }
        else { hN = Math.min(Math.max(hN, 3), 8.2); }
        normalized.height = hN;
      }
      let wN = parseFloat(normalized.weight);
      if (!isNaN(wN)) {
        if (normalized.weightUnit === 'kg') { wN = Math.min(Math.max(wN, 20), 300); }
        else { wN = Math.min(Math.max(wN, 44), 660); }
        normalized.weight = wN;
      }
    } catch {}

    setIsLoading(true);
    try {
      const user = auth.currentUser;
      if (!user) {
        alert('You must be logged in to save your profile.');
        setIsLoading(false);
        return;
      }
  await saveUserProfile(user.uid, { ...normalized, bmi });
      alert('Profile saved successfully!');
    } catch (error) {
      alert('Error saving profile: ' + error.message);
    }
    setIsLoading(false);
  };

  const isFormValid = formData.name && 
                      formData.email && 
                      formData.age && 
                      formData.gender && 
                      formData.height && 
                      formData.weight &&
                      !validationErrors.age && 
                      !validationErrors.height && 
                      !validationErrors.weight;

  return (
    <div className="min-h-screen bg-primary-50 dark:bg-gray-900 py-12 transition-all duration-300">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl sm:text-4xl font-bold text-primary-700 dark:text-primary-400 mb-4">
            <T>Your Health Profile</T>
          </h1>
          <p className="text-lg text-neutral-600 dark:text-neutral-300">
            <T>Help us personalize your diabetes management experience</T>
          </p>
        </div>

        {/* Form */}
        <div className="bg-white/70 dark:bg-gray-800/70 backdrop-blur-sm rounded-2xl shadow-soft border border-neutral-100 dark:border-neutral-700 p-8 transition-all duration-300">
          <form onSubmit={handleSubmit} className="space-y-8">
            
            {/* Personal Information */}
            <div className="space-y-6">
              <h2 className="text-xl font-semibold text-neutral-900 dark:text-white border-b border-neutral-200 dark:border-neutral-600 pb-2 flex items-center">
                <UserIcon className="h-6 w-6 mr-2 text-primary-600 dark:text-primary-400" />
                <T>Personal Information</T>
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    <T>Full Name *</T>
                  </label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 rounded-xl border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-gray-700 text-neutral-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
                    placeholder="Enter your full name"
                    required
                    disabled={!isEditing}
                  />
                </div>

                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    <T>Email Address *</T>
                  </label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 rounded-xl border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-gray-700 text-neutral-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
                    placeholder="Enter your email"
                    required
                    disabled
                  />
                </div>

                {/* Language Preferences
                <div className="bg-primary-50 dark:bg-primary-900/20 rounded-xl p-4 border border-primary-200 dark:border-primary-800">
                  <h4 className="text-sm font-medium text-primary-700 dark:text-primary-300 mb-3 flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                    </svg>
                    <T>Language Preferences</T>
                  </h4>
                  <p className="text-xs text-primary-600 dark:text-primary-400 mb-3">
                    You can also change language using the globe icon (🌐) in the top navigation bar.
                  </p>
                  <LanguageSwitcher className="w-full" />
                </div> */}

                <div>
                  <label htmlFor="age" className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    <T>Age *</T>
                  </label>
                  <input
                    type="number"
                    id="age"
                    name="age"
                    value={formData.age}
                    onChange={handleInputChange}
                    onBlur={handleBlur}
                    className={`w-full px-4 py-3 rounded-xl border ${
                      validationErrors.age 
                        ? 'border-danger-500 focus:ring-danger-500' 
                        : 'border-neutral-300 dark:border-neutral-600 focus:ring-primary-500'
                    } bg-white dark:bg-gray-700 text-neutral-900 dark:text-white focus:ring-2 focus:border-transparent transition-all duration-200`}
                    placeholder="Your age"
                    min="10"
                    max="120"
                    required
                    disabled={!isEditing}
                  />
                  {validationErrors.age && (
                    <p className="mt-1 text-sm text-danger-600 dark:text-danger-400">
                      {validationErrors.age}
                    </p>
                  )}
                </div>

                <div>
                  <label htmlFor="gender" className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    <T>Gender *</T>
                  </label>
                  <select
                    id="gender"
                    name="gender"
                    value={formData.gender}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 rounded-xl border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-gray-700 text-neutral-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
                    required
                    disabled={!isEditing}
                  >
                    <option value=""><T>Select gender</T></option>
                    <option value="male"><T>Male</T></option>
                    <option value="female"><T>Female</T></option>
                    <option value="other"><T>Other</T></option>
                  </select>
                </div>

                <div>
                  <label htmlFor="diabetes_type" className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    <T>Diabetes Type *</T>
                  </label>
                  <select
                    id="diabetes_type"
                    name="diabetes_type"
                    value={formData.diabetes_type}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 rounded-xl border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-gray-700 text-neutral-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
                    required
                    disabled={!isEditing}
                  >
                    <option value="Type1">Type 1</option>
                    <option value="Type2">Type 2</option>
                    <option value="Gestational">Gestational</option>
                    <option value="Prediabetes">Prediabetes</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Physical Measurements */}
            <div className="space-y-6">
              <h2 className="text-xl font-semibold text-neutral-900 dark:text-white border-b border-neutral-200 dark:border-neutral-600 pb-2 flex items-center">
                <ScaleIcon className="h-6 w-6 mr-2 text-primary-600 dark:text-primary-400" />
                <T>Physical Measurements</T>
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    <T>Height *</T>
                  </label>
                  <div className="flex space-x-2">
                    <input
                      type="number"
                      name="height"
                      value={formData.height}
                      onChange={handleInputChange}
                      onBlur={handleBlur}
                      className={`flex-1 px-4 py-3 rounded-xl border ${
                        validationErrors.height 
                          ? 'border-danger-500 focus:ring-danger-500' 
                          : 'border-neutral-300 dark:border-neutral-600 focus:ring-primary-500'
                      } bg-white dark:bg-gray-700 text-neutral-900 dark:text-white focus:ring-2 focus:border-transparent transition-all duration-200`}
                      placeholder={`Height (${formData.heightUnit})`}
                      min={formData.heightUnit === 'cm' ? 100 : 3}
                      max={formData.heightUnit === 'cm' ? 250 : 8.2}
                      step="0.1"
                      required
                      disabled={!isEditing}
                    />
                    <select
                      name="heightUnit"
                      value={formData.heightUnit}
                      onChange={handleInputChange}
                      className="px-3 py-3 rounded-xl border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-gray-700 text-neutral-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
                      disabled={!isEditing}
                    >
                      <option value="cm">cm</option>
                      <option value="ft">ft</option>
                    </select>
                  </div>
                  {validationErrors.height && (
                    <p className="mt-1 text-sm text-danger-600 dark:text-danger-400">
                      {validationErrors.height}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    <T>Weight *</T>
                  </label>
                  <div className="flex space-x-2">
                    <input
                      type="number"
                      name="weight"
                      value={formData.weight}
                      onChange={handleInputChange}
                      onBlur={handleBlur}
                      className={`flex-1 px-4 py-3 rounded-xl border ${
                        validationErrors.weight 
                          ? 'border-danger-500 focus:ring-danger-500' 
                          : 'border-neutral-300 dark:border-neutral-600 focus:ring-primary-500'
                      } bg-white dark:bg-gray-700 text-neutral-900 dark:text-white focus:ring-2 focus:border-transparent transition-all duration-200`}
                      placeholder={`Weight (${formData.weightUnit})`}
                      min={formData.weightUnit === 'kg' ? 20 : 44}
                      max={formData.weightUnit === 'kg' ? 300 : 660}
                      step="0.1"
                      required
                      disabled={!isEditing}
                    />
                    <select
                      name="weightUnit"
                      value={formData.weightUnit}
                      onChange={handleInputChange}
                      className="px-3 py-3 rounded-xl border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-gray-700 text-neutral-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
                      disabled={!isEditing}
                    >
                      <option value="kg">kg</option>
                      <option value="lbs">lbs</option>
                    </select>
                  </div>
                  {validationErrors.weight && (
                    <p className="mt-1 text-sm text-danger-600 dark:text-danger-400">
                      {validationErrors.weight}
                    </p>
                  )}
                </div>
              </div>

              {/* BMI Display */}
              {bmi !== null && (
                <div className="mt-4 p-4 bg-primary-50 dark:bg-primary-900/20 rounded-xl">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-neutral-600 dark:text-neutral-400"><T>Your BMI</T></p>
                      <p className="text-2xl font-bold text-neutral-900 dark:text-white">{bmi}</p>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-sm font-medium ${bmiCategory.color}`}>
                      {bmiCategory.category}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Health Information */}
            {/* Submit Button */}
            <div className="pt-6">
              <button
                type="submit"
                disabled={!isFormValid || isLoading || !isEditing}
                className={`
                  w-full flex items-center justify-center px-6 py-4 rounded-xl font-semibold text-lg transition-all duration-200
                  ${isFormValid && !isLoading && isEditing
                    ? 'bg-primary-600 hover:bg-primary-700 text-white shadow-medium hover:shadow-strong transform hover:-translate-y-1'
                    : 'bg-neutral-300 dark:bg-neutral-700 text-neutral-500 dark:text-neutral-400 cursor-not-allowed'
                  }
                `}
              >
                {isLoading ? (
                  <>
                    <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
                    <T>Saving Profile...</T>
                  </>
                ) : (
                  <>
                    <CheckIcon className="h-5 w-5 mr-2" />
                    <T>Save Profile</T>
                  </>
                )}
              </button>
              <button
                type="button"
                className="mt-4 w-full flex items-center justify-center px-6 py-3 rounded-xl font-semibold text-base bg-primary-100 dark:bg-primary-900/20 text-primary-700 dark:text-primary-400 hover:bg-primary-200 dark:hover:bg-primary-900/40 transition-all duration-200"
                onClick={() => setIsEditing((prev) => !prev)}
              >
                <T>{isEditing ? 'Cancel' : 'Edit'}</T>
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Profile;
