/**
 * screenshot-slides.mjs
 *
 * Captura screenshots de cada slide de uma apresentação Slidev (build de produção)
 * usando Playwright. Contorna o bug do `slidev export` onde Vue não monta em CI
 * (body fica display:none indefinidamente).
 *
 * Uso: node scripts/screenshot-slides.mjs <baseUrl> <outDir> <totalSlides>
 * Ex:  node scripts/screenshot-slides.mjs http://localhost:8080/ /tmp/slides-lec01 45
 */

import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const [,, baseUrl, outDir, totalStr] = process.argv;

if (!baseUrl || !outDir) {
  console.error('Uso: screenshot-slides.mjs <baseUrl> <outDir> [totalSlides]');
  process.exit(1);
}

const total = parseInt(totalStr) || 80;
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({
  args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
});

let exported = 0;

for (let i = 1; i <= total; i++) {
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1920, height: 1080 });

  // Injeta CSS para forçar body visível ANTES da navegação
  await page.addInitScript(() => {
    const s = document.createElement('style');
    s.textContent = 'body,html{display:block!important;visibility:visible!important;opacity:1!important}';
    document.addEventListener('DOMContentLoaded', () => {
      if (s.parentNode === null) document.head.appendChild(s);
    });
  });

  await page.goto(`${baseUrl}${i}`, {
    waitUntil: 'networkidle',
    timeout: 25000
  }).catch(() => {});

  // Detecta loop: Slidev redireciona para slide 1 quando ultrapassa o total
  if (i > 2) {
    const url = page.url();
    if (url.endsWith('/1') || url.endsWith('/1/')) {
      console.log(`\n  Fim detectado após slide ${i - 1}`);
      await page.close();
      break;
    }
  }

  // Força visibilidade após o carregamento
  await page.addStyleTag({
    content: 'body,html{display:block!important;visibility:visible!important;opacity:1!important}'
  }).catch(() => {});
  await page.evaluate(() => {
    try {
      document.body.style.cssText += ';display:block!important;visibility:visible!important;opacity:1!important';
    } catch (_) {}
  });

  await page.waitForTimeout(900);

  const num = String(i).padStart(3, '0');
  await page.screenshot({
    path: `${outDir}/slide-${num}.png`,
    clip: { x: 0, y: 0, width: 1920, height: 1080 }
  });

  exported++;
  process.stdout.write(`  slide ${i}/${total}\r`);
  await page.close();
}

await browser.close();
console.log(`\nConcluído: ${exported} slides capturados em ${outDir}`);
