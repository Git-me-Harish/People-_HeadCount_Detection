import i18n from "i18next";
import { initReactI18next } from "react-i18next";

const resources = {
  en: {
    translation: {
      // Navigation
      dashboard: "Dashboard",
      cameras: "Cameras",
      liveStream: "Live Stream",
      analytics: "Analytics",
      alerts: "Alerts",
      notifications: "Notifications",
      reports: "Reports",
      settings: "Settings",
      apiTokens: "API Tokens",
      auditLog: "Audit Log",
      plan: "Plan & Usage",
      publicPage: "Public Page",
      signOut: "Sign Out",

      // Common
      save: "Save",
      cancel: "Cancel",
      delete: "Delete",
      edit: "Edit",
      create: "Create",
      loading: "Loading…",
      error: "Something went wrong",
      noData: "No data available",
      search: "Search",
      active: "Active",
      inactive: "Inactive",
      enabled: "Enabled",
      disabled: "Disabled",

      // Dashboard
      currentCount: "Current Count",
      peakToday: "Peak Today",
      avgCount: "Average Count",
      totalDetections: "Total Detections",
      liveCameras: "Live Cameras",
      recentAlerts: "Recent Alerts",

      // Onboarding
      welcomeToPeopleSense: "Welcome to PeopleSense",
      chooseVertical: "Choose your industry vertical to get started",
      applyTemplate: "Apply Template",
      templateApplied: "Template applied successfully",
      skipOnboarding: "Skip and set up manually",

      // Cameras
      addCamera: "Add Camera",
      cameraName: "Camera Name",
      cameraLocation: "Location",
      streamUrl: "Stream URL",
      noCameras: "No cameras configured yet",

      // Alerts
      addAlert: "Add Alert",
      alertName: "Alert Name",
      threshold: "Threshold (people)",
      webhookUrl: "Webhook URL",
      noAlerts: "No alert rules configured",
      lastTriggered: "Last triggered",
      never: "Never",

      // Notifications
      markAllRead: "Mark all read",
      noNotifications: "All caught up!",
      unreadNotifications: "{{count}} unread",

      // Plan
      freePlan: "Free Plan",
      proPlan: "Pro Plan",
      enterprisePlan: "Enterprise Plan",
      camerasUsed: "Cameras Used",
      framesThisMonth: "Frames This Month",
      alertsSentThisMonth: "Alerts Sent",
      upgradePlan: "Upgrade Plan",

      // Auth
      signIn: "Sign In",
      signUp: "Sign Up",
      email: "Email",
      password: "Password",
      fullName: "Full Name",
      organizationName: "Organisation Name",
      forgotPassword: "Forgot password?",
      alreadyHaveAccount: "Already have an account?",
      dontHaveAccount: "Don't have an account?",
    },
  },
  hi: {
    translation: {
      // Navigation
      dashboard: "डैशबोर्ड",
      cameras: "कैमरे",
      liveStream: "लाइव स्ट्रीम",
      analytics: "विश्लेषण",
      alerts: "अलर्ट",
      notifications: "सूचनाएं",
      reports: "रिपोर्ट",
      settings: "सेटिंग्स",
      apiTokens: "API टोकन",
      auditLog: "ऑडिट लॉग",
      plan: "योजना और उपयोग",
      publicPage: "सार्वजनिक पृष्ठ",
      signOut: "साइन आउट",

      // Common
      save: "सहेजें",
      cancel: "रद्द करें",
      delete: "हटाएं",
      edit: "संपादित करें",
      create: "बनाएं",
      loading: "लोड हो रहा है…",
      error: "कुछ गलत हो गया",
      noData: "डेटा उपलब्ध नहीं",
      search: "खोजें",
      active: "सक्रिय",
      inactive: "निष्क्रिय",
      enabled: "सक्षम",
      disabled: "अक्षम",

      // Dashboard
      currentCount: "वर्तमान गिनती",
      peakToday: "आज का शिखर",
      avgCount: "औसत गिनती",
      totalDetections: "कुल पहचान",
      liveCameras: "लाइव कैमरे",
      recentAlerts: "हाल के अलर्ट",

      // Onboarding
      welcomeToPeopleSense: "PeopleSense में आपका स्वागत है",
      chooseVertical: "शुरू करने के लिए अपना उद्योग चुनें",
      applyTemplate: "टेम्पलेट लागू करें",
      templateApplied: "टेम्पलेट सफलतापूर्वक लागू किया गया",
      skipOnboarding: "छोड़ें और मैन्युअल रूप से सेट करें",

      // Auth
      signIn: "साइन इन",
      signUp: "साइन अप",
      email: "ईमेल",
      password: "पासवर्ड",
      fullName: "पूरा नाम",
      organizationName: "संगठन का नाम",
    },
  },
};

i18n.use(initReactI18next).init({
  resources,
  lng: localStorage.getItem("ps_language") ?? "en",
  fallbackLng: "en",
  interpolation: { escapeValue: false },
});

export default i18n;
