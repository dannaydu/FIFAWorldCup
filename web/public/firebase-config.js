// Firebase WEB config — PUBLIC by design (these identify the project; security
// is enforced by Firestore rules, not by hiding these values). Safe to commit.
//
// Used only for optional Google sign-in / cloud-saved fantasy cards (see auth.js).
// Forecast/results/edges data always loads from the deployed JSON snapshots.
window.FIREBASE_CONFIG = {
  apiKey: "AIzaSyCS7GJzJbsoMT9FN3rG5Z_X9cgkBD0vjnU",
  authDomain: "fifa-3b360.firebaseapp.com",
  projectId: "fifa-3b360",
  storageBucket: "fifa-3b360.firebasestorage.app",
  messagingSenderId: "1011104722574",
  appId: "1:1011104722574:web:06a940bb36285a078f63df",
  measurementId: "G-0XQCGFRGWZ",
};
