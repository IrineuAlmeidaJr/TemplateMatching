/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './public/libs/flowbite/**/*.js',
  ],
  theme: {
    fontFamily: {
      sans: [
        "Inter var, sans-serif"]
    },
  },
  plugins: [
    require('./public/libs/flowbite/plugin.js')
  ]
}
