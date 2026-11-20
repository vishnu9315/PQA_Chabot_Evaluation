import { defineConfig, devices } from '@playwright/test';
import dotenv from 'dotenv';

dotenv.config();

/**
 * One-off config for scripts/capture-auth.ts - run via `npm run login` (or
 * `npm run auth:capture` with PQA_IDENTITY_EMAIL set). PQA_BASE_URL comes from .env.
 *
 * Headed, single-worker, no retries: a human relays a real emailed OTP mid-run (see
 * scripts/capture-auth.ts), so parallelism or auto-retry here would just race multiple
 * OTP requests against the same inbox.
 */
export default defineConfig({
  testDir: '.',
  testMatch: /capture-auth\.ts/,
  fullyParallel: false,
  retries: 0,
  workers: 1,
  timeout: 5 * 60_000,
  use: {
    baseURL: process.env.PQA_BASE_URL ?? 'https://pqa.myscheme.in/meityy',
    // Headless by default: flip to false locally if you want to watch it run. Nothing
    // about the OTP relay depends on a visible window - the file-based handoff in
    // scripts/capture-auth.ts works identically either way.
    headless: true,
    // Deliberately short: without this, a wrong selector just silently hangs until the
    // overall 5-minute test timeout (which exists for the OTP wait, not ordinary
    // actions) - that makes a real bug indistinguishable from "still waiting for the
    // OTP" for minutes at a time.
    actionTimeout: 15_000,
    navigationTimeout: 20_000,
    trace: 'on',
    screenshot: 'on',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
