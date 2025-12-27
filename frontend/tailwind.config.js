/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'honeywell-dark': '#1a1a1a',
                'honeywell-cream': '#fcfbf4',
                'honeywell-blue': '#005EB8', // Example Honeywell blue
                'honeywell-red': '#ff4b4b',
            }
        },
    },
    plugins: [],
}
