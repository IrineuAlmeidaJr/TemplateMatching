import BoxInput from "@/components/BoxInput";
import BoxShowImage from "@/components/BoxShowImage";
import Header from "@/components/Header";

import { useGlobalContext } from "@/context/Global";
import ShowReport from "../components/ShowReport";

export default function Home() {
  const  globalContext = useGlobalContext();

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
        <div className="flex mb-4">         
          <BoxInput />
          
          <BoxShowImage />
        </div>

        {
          globalContext.showReport && 
          <ShowReport />
        }

      </div>

      
    </main>
  )
}
