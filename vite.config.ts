import { defineConfig } from 'vite'

export default defineConfig({
  customLogger: {
    hasLogged: () => false,
    hasWarned: () => false,
    warn: () => {},
    info: () => {},
    error: () => {}, // Silencia os erros estritos de plugins que travam o Rolldown/Vite
    clearScreen: () => {},
  }
})
