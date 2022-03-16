{
int numberOfImages = 5000;
std::vector<TCanvas*> canvases(numberOfImages);
for (int i=0; i<numberOfImages; i++) {
  TCanvas *c = new TCanvas(("c" + std::to_string(i)).c_str(), std::to_string(i).c_str(),700,600);
  canvases[i] = c;
  c->SetLogz();
  gStyle->SetOptStat(0);
  TH2F *liquido_event = new TH2F("liquido_event","liquido_event",200,-100,100,200,-100,100);
  liquido_event->SetTickLength(0.,"X");
  liquido_event->SetTickLength(0.,"Y");
  liquido_event->SetLabelSize(0.,"X");
  liquido_event->SetLabelSize(0.,"Y");
  liquido_event->SetTitle("");
  std:string entry = "Entry$==";
  entry += std::to_string(i);
  op_hits->Draw("h_pos_y/10.:h_pos_x/10.>>liquido_event",entry.c_str(),"col");
  
  std::string filenameTest = "/home/kylesinger/simulation/build/images/Test/Positron/img_"+std::to_string(i)+".jpg";
  c->SaveAs(filenameTest.c_str());
  delete c;
  delete liquido_event;
 }
}

