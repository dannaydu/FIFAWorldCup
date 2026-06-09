// Firebase web config.
//
// LOCAL/STATIC MODE (default): leave apiKey as "REPLACE_ME". The app reads the
// bundled JSON in /data and works immediately once hosted — no credentials.
//
// LIVE MODE: paste your Firebase web app config here (Firebase console ->
// Project settings -> General -> Your apps -> Web app -> SDK setup). The app
// will then read live from Firestore (collection "snapshots") instead of JSON.
window.FIREBASE_CONFIG = {
  apiKey: "REPLACE_ME",
  authDomain: "REPLACE_ME.firebaseapp.com",
  projectId: "REPLACE_ME",
  storageBucket: "REPLACE_ME.appspot.com",
  messagingSenderId: "REPLACE_ME",
  appId: "REPLACE_ME",
};
