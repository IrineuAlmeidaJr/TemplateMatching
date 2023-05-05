

export default function Header({ title }) {

    return (
        <header className="px-4 py-8 flex justify-center items-center">
            <h1
            className="
            font-extrabold
            leading-tight 
            tracking-tight 
            text-[#B6C3D4]
            text-2xl 
            text-center
            uppercase">
                { title } 
            </h1>
        </header>
    )
}