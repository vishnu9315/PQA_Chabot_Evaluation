import path from 'node:path';
import type { Page } from '@playwright/test';

/**
 * TEMPLATE: genericized from a previous project. That project's single login identity
 * could switch between several roles/divisions post-login (via an in-app "Switch
 * Profile" picker), and captured one separate storageState snapshot per role - because
 * the chatbot endpoint's retrieved documents/behavior could depend on which role was
 * active when the snapshot was taken, not just on prompt text.
 *
 * Below is ONE default workspace that does no role-switching at all - the common case if
 * your app has a single role per login. If your app DOES have a role/division/workspace
 * picker you need to select after login, add more entries here (see the commented
 * example) and capture-auth.ts will save one storageState file per entry automatically.
 */
const authDir = path.resolve(__dirname, '..', '.auth');

export type WorkspaceName = 'default';

export interface WorkspaceConfig {
  name: WorkspaceName;
  storageStatePath: string;
  /** Runs after login, before saving this workspace's storageState. No-op by default. */
  selectInApp: (page: Page) => Promise<void>;
}

export function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(
      `Missing required environment variable: ${name}. Run \`npm run login\`, which asks for it.`);
  }
  return value;
}

export function getWorkspaces(): Record<WorkspaceName, WorkspaceConfig> {
  return {
    default: {
      name: 'default',
      storageStatePath: path.join(authDir, 'default.json'),
      selectInApp: async () => {
        // No role/division picker by default - nothing to do here. If your app needs
        // one, replace this with the equivalent of the commented example below.
      },
    },

    // --- Example: a role/division picker, like the previous project had -------------
    // Uncomment and adapt if your app needs it. Also add the new name to WorkspaceName
    // above, e.g. 'default' | 'exportDivision'.
    //
    // exportDivision: {
    //   name: 'exportDivision',
    //   storageStatePath: path.join(authDir, 'export-division.json'),
    //   selectInApp: async (page) => {
    //     await page.getByRole('button', { name: 'Account menu' }).click();
    //     await page.getByRole('menuitem', { name: 'Switch Profile' }).click();
    //     await page.getByRole('button', { name: 'Department User' }).click();
    //     await page.getByRole('button', { name: /Export Division/ }).click();
    //     await page.getByRole('button', { name: 'Account menu' }).waitFor({ timeout: 45_000 });
    //   },
    // },
  };
}
