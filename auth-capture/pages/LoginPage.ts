import { expect, type Page } from '@playwright/test';

/**
 * Email + emailed-OTP sign-in - no "dev sign-in" shortcut.
 *
 * Verified end to end against the MeitY portal (https://pqa.myscheme.in/meityy) on
 * 2026-11-20: email, "Send code", the emailed 6-digit code, "Verify", then the signed-in
 * page. If the portal's login form changes, this is the file to update; compare with
 * `npx playwright codegen https://pqa.myscheme.in/meityy`.
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
    // The MeitY login form lives at the full PQA_BASE_URL (https://pqa.myscheme.in/meityy).
    // goto('/') would resolve to the bare origin and miss the /meityy path, so navigate to
    // the full URL explicitly.
    await this.page.goto(process.env.PQA_BASE_URL ?? 'https://pqa.myscheme.in/meityy');
    await this.emailInput.fill(email);
    await this.sendCodeButton.click();
    await this.codeInput.waitFor({ timeout: 15_000 });
  }

  async enterCode(code: string) {
    await this.codeInput.fill(code);
    await this.verifyButton.click();
    // After verification the portal shows either a role picker or the signed-in page.
    await Promise.race([
      this.page.getByText('Select your role to continue').waitFor({ timeout: 45_000 }),
      this.page.getByRole('button', { name: 'Account menu' }).waitFor({ timeout: 45_000 }),
    ]);
  }

  async expectSignedOut() {
    await expect(this.emailInput).toBeVisible();
  }
}
