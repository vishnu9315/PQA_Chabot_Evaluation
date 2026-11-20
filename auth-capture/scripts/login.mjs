#!/usr/bin/env node
// Interactive portal login: asks for your email, runs the Playwright capture, then asks for
// the emailed 6-digit code and hands it to the capture (via .auth/.otp-relay.txt).
//
//   npm run login                         prompts for email and code
//   npm run login -- --email you@gov.in   prompts only for the code
//   npm run login -- --dry-run            shows what would run, sends nothing
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const relayPath = path.join(root, '.auth', '.otp-relay.txt');
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};
const dryRun = args.includes('--dry-run');

const rl = readline.createInterface({ input: process.stdin });
const lines = rl[Symbol.asyncIterator]();
async function ask(question) {
  process.stdout.write(question);
  const { value, done } = await lines.next();
  if (done) {
    console.error('\nNo input received.');
    process.exit(2);
  }
  return value.trim();
}

async function askUntil(question, isValid, hint) {
  for (;;) {
    const answer = await ask(question);
    if (isValid(answer)) return answer;
    console.log(hint);
  }
}

let email = flag('--email');
if (email && !EMAIL_RE.test(email)) {
  console.error(`"${email}" is not a valid email address.`);
  process.exit(2);
}
email ??= await askUntil('Portal login email: ', (a) => EMAIL_RE.test(a),
  'Enter a full email address, e.g. name@example.gov.in');

const command = 'npx playwright test --config=playwright.auth.config.ts';
if (dryRun) {
  console.log(`Would run: ${command}`);
  console.log(`  with PQA_IDENTITY_EMAIL=${email}`);
  console.log(`  and write the code you enter to ${relayPath}`);
  rl.close();
  process.exit(0);
}

fs.rmSync(relayPath, { force: true });
const child = spawn(command, {
  cwd: root,
  shell: true,
  env: { ...process.env, PQA_IDENTITY_EMAIL: email },
  stdio: ['ignore', 'pipe', 'inherit'],
});

let asked = false;
let seen = '';
child.stdout.on('data', async (chunk) => {
  const text = chunk.toString();
  process.stdout.write(text);
  seen = (seen + text).slice(-2000);
  if (!asked && seen.includes('OTP requested')) {
    asked = true;
    const code = await askUntil('Enter the 6-digit code from your email: ',
      (a) => /^\d{6}$/.test(a), 'The code is 6 digits.');
    fs.writeFileSync(relayPath, code);
  }
});

child.on('exit', (codeOut) => {
  rl.close();
  if (codeOut === 0) {
    console.log('Login saved to auth-capture/.auth/default.json. It lasts about 30 minutes.');
  }
  process.exit(codeOut ?? 1);
});
