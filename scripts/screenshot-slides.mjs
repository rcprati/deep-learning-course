/**
 * screenshot-slides.mjs
 *
 * Captura screenshots de cada slide usando ArrowRight para navegação.
 * Não depende de window.__slidev__ — usa contagem de '---' no .md como total.
 * Injeta CSS !important antes dos scripts da página para impedir body { display:none }.
 *
 * Uso: node scripts/screenshot-slides.mjs <baseUrl> <outDir> <arquivo.md>
 */

import { chromium } from 'playwright';
import { mkdirSync, readFileSync } from 'fs';

const [,, baseUrl, outDir, mdFile] = process.argv;

if (!baseUrl || !outDir) {
  console.error('Uso: screenshot-slides.mjs <baseUrl> <outDir> <arquivo.md>');
  process.exit(1);
}

mkdirSync(outDir, { recursive: true });

// Conta slides: cada '---' após o frontmatter é um separador
let total = 80;
if (mdFile) {
  try {
    const content = readFileSync(mdFile, 'utf8');
    const seps = (content.match(/^---$/gm) || []).length;
    total = Math.max(seps - 1, 1);
    console.log(`Slides (do .md): ${total}`);
  } catch (_) {
    console.log(`Slides (padrão): ${total}`);
  }
}

const FORCE_CSS = 'html,body{display:block!important;visibility:visible!important;opacity:1!important}';

const browser = await chromium.launch({
  args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
});

const page = await browser.newPage();
await page.setViewportSize({ width: 1920, height: 1080 });

// addInitScript roda ANTES de qualquer script da página.
// CSS !important em stylesheet sobrepõe inline style (display:none) do Slidev.
await page.addInitScript(() => {
  const css = 'html,body{display:block!important;visibility:visible!important;opacity:1!important}';
  const inject = () => {
    const s = document.createElement('style');
    s.textContent = css;
    (document.head || document.documentElement).appendChild(s);
  };
  inject();
  document.addEventListener('DOMContentLoaded', inject);
  window.addEventListener('load', inject);
});

console.log(`Carregando ${baseUrl} ...`);
await page.goto(baseUrl, { waitUntil: 'domcontentloaded', timeout: 30000 });

// Reinjeta após navegação (garante que sobrescreveu o inline style do Vue)
await page.addStyleTag({ content: FORCE_CSS });
await page.waitForTimeout(4000);   // aguarda Slidev inicializar e renderizar slide 1
await page.addStyleTag({ content: FORCE_CSS });

// Garante foco para receber eventos de teclado
await page.mouse.click(960, 540);
await page.waitForTimeout(300);

for (let i = 1; i <= total; i++) {
  // Reinjeta CSS a cada slide (Slidev pode re-esconder em transições)
  await page.addStyleTag({ content: FORCE_CSS }).catch(() => {});
  await page.waitForTimeout(650);

  const num = String(i).padStart(3, '0');
  await page.screenshot({
    path: `${outDir}/slide-${num}.png`,
    clip: { x: 0, y: 0, width: 1920, height: 1080 }
  });
  process.stdout.write(`  ${i}/${total}\r`);

  if (i < total) {
    await page.keyboard.press('ArrowRight');
  }
}

await browser.close();
console.log(`\nConcluído: ${total} slides → ${outDir}`);
