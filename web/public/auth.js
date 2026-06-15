// Optional accounts: Google sign-in + Firestore-synced fantasy cards.
//
// Activates ONLY when web/public/firebase-config.js is filled in AND you've
// enabled Authentication (Google) + Firestore in the Firebase console. Until
// then `AUTH.available` is false and the app runs 100% on localStorage — no
// sign-in UI, nothing breaks.
window.AUTH = (() => {
  const cfg = window.FIREBASE_CONFIG || {};
  const available = !!cfg.apiKey && cfg.apiKey !== "REPLACE_ME";
  let user = null, auth = null, db = null, mods = null, ready = null;
  const listeners = [];

  async function ensure() {
    if (!available) return null;
    if (!ready) {
      ready = (async () => {
        const appMod = await import("https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js");
        const authMod = await import("https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js");
        const fsMod = await import("https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js");
        const app = appMod.getApps().length ? appMod.getApp() : appMod.initializeApp(cfg);
        auth = authMod.getAuth(app);
        db = fsMod.getFirestore(app);
        mods = { authMod, fsMod };
        authMod.onAuthStateChanged(auth, (u) => { user = u; listeners.forEach((cb) => cb(u)); });
        return true;
      })().catch((e) => { console.warn("auth init failed", e); return false; });
    }
    return ready;
  }

  async function signIn() {
    if (!(await ensure())) return;
    try {
      await mods.authMod.signInWithPopup(auth, new mods.authMod.GoogleAuthProvider());
    } catch (e) { console.warn("sign-in failed", e); }
  }
  async function signOut() { if (await ensure()) await mods.authMod.signOut(auth); }

  async function loadCard() {
    if (!(await ensure()) || !user) return null;
    const { doc, getDoc } = mods.fsMod;
    try {
      const snap = await getDoc(doc(db, "fantasy", user.uid));
      return snap.exists() ? snap.data() : null;
    } catch { return null; }
  }
  async function saveCard(picks) {
    if (!(await ensure()) || !user) return;
    const { doc, setDoc } = mods.fsMod;
    try { await setDoc(doc(db, "fantasy", user.uid), { picks, updated: Date.now() }); }
    catch (e) { console.warn("cloud save failed", e); }
  }

  function onChange(cb) { listeners.push(cb); if (available) ensure(); }
  const currentUser = () => user;

  return { available, signIn, signOut, loadCard, saveCard, onChange, currentUser };
})();
