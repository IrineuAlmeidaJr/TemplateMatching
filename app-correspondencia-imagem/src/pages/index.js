import BoxInput from "@/components/BoxInput";
import BoxShowImage from "@/components/BoxShowImage";
import Header from "@/components/Header";

export default function Home() {
  return (
    <main className="flex justify-center items-center">
      <div 
      className="
      my-10 
      min-w-[1000px]
      w-[1000px]  
      bg-[#F3F4F9] 
      rounded-lg">

        <Header title="Correspondência de Imagem"/>
        
        <div className="flex w-full mb-4">         
          <BoxInput />
          
          <BoxShowImage />
        </div>

      </div>

      
    </main>
  )
}
