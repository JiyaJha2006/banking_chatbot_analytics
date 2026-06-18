import { test, expect } from '@playwright/test';

test('full banking chatbot button flow', async ({ page }) => {
  await page.goto('http://127.0.0.1:8000/');

  await page.getByRole('button', { name: 'Sign up', exact: true }).click();

  await page.getByRole('textbox', { name: 'Username' }).fill('Om' + Date.now());
  await page.getByRole('textbox', { name: 'Password' }).fill('password');

  await page.getByRole('button', { name: 'Create account' }).click();

  await page.getByRole('button', { name: 'Hindi' }).click();
  await page.getByRole('button', { name: 'English' }).click();

  await page.getByRole('button', { name: 'Profile' }).click();

  await page.getByRole('button', { name: 'Chat', description: 'Chat', exact: true }).click();

  await page.getByRole('button', { name: 'Collapse sidebar' }).click();
  await page.getByRole('button', { name: 'Expand sidebar' }).click();

  await expect(page.getByRole('button', { name: 'Voice' })).toBeVisible();

  await page.getByRole('textbox', { name: 'Ask a banking question' }).fill('What is a savings account?');
  await page.locator('form').getByRole('button').filter({ hasText: /^$/ }).click();

  await expect(
    page.getByRole('paragraph').filter({ hasText: /savings account/i })
  ).toBeVisible();

  await page.getByRole('button', { name: 'Profile' }).click();

  await page.getByPlaceholder('Search chats').fill('savings');
  await page.getByRole('button', { name: /What is a savings account/i }).click();

  await page.getByRole('button', { name: 'Chat', description: 'Chat', exact: true }).click();

  await page.getByPlaceholder('Search chats').fill('');

  await page.getByRole('button', { name: 'New chat' }).click();

  await page.getByPlaceholder('Search chats').fill('sav');
  await page.getByPlaceholder('Search chats').press('Enter');

  await page.getByRole('button', { name: 'Log out' }).click();
});