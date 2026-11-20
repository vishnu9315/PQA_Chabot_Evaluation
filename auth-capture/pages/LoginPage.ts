import { expect, type Page } from '@playwright/test';

/**
 * Email + emailed-OTP sign-in - no "dev sign-in" shortcut.
 *
 * TEMPLATE: copied from a previous project on what you said is the same login flow
 * (email, then a real emailed 6-digit code). The selectors below (accessible names like
 * "Sign-in email", "Send code", etc.) matched that project's app exactly; verify they
 * still match yours by opening your login page with Playwright's codegen
 * (`npx playwright codegen <your-base-url>`) and comparing, before trusting a real run.
 * If your app's field names differ, this is the one file you most likely need to edit.
 *
 * Only used by scripts/capture-auth.ts.
 */
export class LoginPage {
  constructor(private readonly page: Page) {}

  private get emailInput() {
    return this.page.getByRole('textbox', { name: 'Sign-in email' });
  }

  private get sendCodeButton() {
    return this.page.getByRole('button', { name: 'Send code' });
  }

  private get codeInput() {
    return this.page.getByRole('textbox', { name: 'Sign-in code' });
  }

  private get verifyButton() {
    return this.page.getByRole('button', { name: 'Verify' });
  }

  async requestCode(email: string) {
    // TEMPLATE: the previous project's app lived behind a "Select a portal" picker at
    // its bare origin, so its login page needed a specific sub-path, not '/'. Check
    // whether your app's real login page is actually at your baseURL's root before
    // assuming '/' is right - a goto('/') here is the most common thing to need fixing.
    await this.page.goto('/');
    await this.emailInput.fill(email);
    await this.sendCodeButton.click();
    await this.codeInput.waitFor({ timeout: 15_000 });
  }

  async enterCode(code: string) {
    await this.codeInput.fill(code);
    await this.verifyButton.click();
    // TEMPLATE: the previous project's identity could hold multiple roles, landing on
    // either a role picker or straight on a dashboard. Adjust this race to whatever your
    // app actually shows right after a successful code verification.
    await Promise.race([
      this.page.getByText('Select your role to continue').waitFor({ timeout: 45_000 }),
      this.page.getByRole('button', { name: 'Account menu' }).waitFor({ timeout: 45_000 }),
    ]);
  }

  async expectSignedOut() {
    await expect(this.emailInput).toBeVisible();
  }
}
