import fs from 'node:fs';
import path from 'node:path';
import { test as setup } from '@playwright/test';
import { LoginPage } from '../pages/LoginPage';
import { getWorkspaces, requireEnv } from '../fixtures/workspaces';

const identityEmail = requireEnv('PQA_IDENTITY_EMAIL');
const workspaces = getWorkspaces();

/**
 * One-time, human-assisted session capture, producing a Playwright storageState JSON per
 * workspace in fixtures/workspaces.ts - for capture_new_chatbot.py's PQA_CHATBOT_SESSION_FILE.
 *
 * TEMPLATE: copied from a previous project with the same login flow you described (email,
 * then a real emailed 6-digit code, no dev-sign-in shortcut) - so this cannot be re-run
 * unattended; it requests a code, then pauses and polls OTP_RELAY_PATH for up to
 * OTP_WAIT_MS. Write the 6-digit code you receive by email into that file (plain text,
 * nothing else) to unblock it.
 *
 * Run via `npm run auth:capture` whenever the saved session(s) expire.
 *
 * One login is shared across every workspace: after verifying the OTP once, the script
 * walks each configured workspace (running its selectInApp, a no-op by default) and saves
 * a separate storageState snapshot per workspace. If your app keeps its active
 * role/division selection in localStorage rather than purely server-side, Playwright's
 * storageState() captures that natively too - no extra handling needed either way.
 */
const OTP_RELAY_PATH = path.join(__dirname, '..', '.auth', '.otp-relay.txt');
const OTP_WAIT_MS = 180_000;

async function waitForOtp(): Promise<string> {
  const deadline = Date.now() + OTP_WAIT_MS;
  while (Date.now() < deadline) {
    if (fs.existsSync(OTP_RELAY_PATH)) {
      const otp = fs.readFileSync(OTP_RELAY_PATH, 'utf-8').trim();
      if (otp) {
        fs.unlinkSync(OTP_RELAY_PATH);
        return otp;
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error(
    `Timed out after ${OTP_WAIT_MS}ms waiting for an OTP. Write the 6-digit code to ` +
      `${OTP_RELAY_PATH} (plain text) and re-run \`npm run auth:capture\`.`,
  );
}

setup('capture a shared session, then one storageState per workspace', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.requestCode(identityEmail);

  // eslint-disable-next-line no-console
  console.log(`[auth-capture] OTP requested for ${identityEmail}.`);
  // eslint-disable-next-line no-console
  console.log(`[auth-capture] Write the 6-digit code to: ${OTP_RELAY_PATH}`);

  const otp = await waitForOtp();
  await loginPage.enterCode(otp);

  for (const workspace of Object.values(workspaces)) {
    await workspace.selectInApp(page);
    fs.mkdirSync(path.dirname(workspace.storageStatePath), { recursive: true });
    await page.context().storageState({ path: workspace.storageStatePath });
    // eslint-disable-next-line no-console
    console.log(`[auth-capture] Saved ${workspace.name} -> ${workspace.storageStatePath}`);
  }
});
