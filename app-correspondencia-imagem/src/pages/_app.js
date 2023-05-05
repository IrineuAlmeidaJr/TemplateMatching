import GlobalContextProvider from '@/context/Global';
import '@/styles/globals.css'

import { initFlowbite } from 'flowbite'
import Head from 'next/head';
import { useEffect } from 'react'

export default function App({ Component, pageProps }) {

  useEffect(() => {
    initFlowbite();
  },[])

  return (
    <GlobalContextProvider>
      <Head>
        <title>Correspondencia de Imagem</title>
      </Head>
      <Component {...pageProps} />
    </GlobalContextProvider>
    
  )
}
